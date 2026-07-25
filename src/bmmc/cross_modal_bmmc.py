"""
Cross-Modal CA vs CP Experiment: RNA (scGPT embeddings) x ADT (MLP)
NeurIPS 2021 BMMC CITE-seq dataset (90,261 cells)

Signal:  Cell type — shared biological identity across RNA and surface protein modalities.
Noise:   Donor effects — biological/technical variation correlated across modalities.

Biological analogy to the astro experiment:
    LAMOST spectra (SpectralViT, frozen)  →  RNA counts (scGPT, frozen)
    Kepler light curves (DualFormer)      →  ADT proteins (MLP, trainable)
    Rotation period (Prot)               →  Cell type
    Telescope systematics                →  Donor effects

Stage 0 (run once):  extract RNA embeddings via scGPT → .npy on disk
Stage 1 (this file): CA / CP training on (frozen RNA embeds) x (trainable ADT MLP)
Evaluation:          cell type balanced_accuracy (signal — want high)
                     donor   balanced_accuracy (noise  — want low)

Usage:
    # Stage 0: precompute RNA embeddings
    python src/cross_modal_bmmc.py --extract_rna_only

    # Stage 1: cross-modal experiment
    python src/cross_modal_bmmc.py --mode all
"""

from __future__ import annotations
import os
import sys
import copy
import argparse
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM_ROOT = Path(__file__).resolve().parent.parent.parent
BMMC_DATA_DIR  = Path('/home/ilay.kamai/work/citeseq/bmmc')

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = {
    'h5ad_path':          str(BMMC_DATA_DIR / 'neurips2021_cite_BMMC.h5ad'),
    'rna_embeddings':     str(BMMC_DATA_DIR / 'rna_embeddings_scgpt.npy'),
    'scgpt_checkpoint':   str(Path('/home/ilay.kamai/work/citeseq') / 'scGPT_human'),
    'n_top_genes':        2000,
    # ADT encoder
    'adt_hidden_dims':    [256, 512],
    'adt_embed_dim':      512,
    # scGPT output dim
    'rna_embed_dim':      512,
    # Output
    'output_dir':         str(MM_ROOT / 'src/logs/bmmc'),
}

SIGNAL_LABELS = ['cell_type']   # high = good
NOISE_LABELS  = ['donor']       # high = bad  (donor leakage)


# ============================================================================
# Stage 0: RNA Embedding Extraction (scGPT)
# ============================================================================

def extract_rna_embeddings(config: dict, rna_adata, device: str = 'cuda') -> np.ndarray:
    """
    Extract scGPT cell embeddings from the HVG RNA AnnData.
    Skips if cached .npy already exists.

    Returns (N, 512) float32 array.
    """
    emb_path = config['rna_embeddings']

    if os.path.exists(emb_path):
        print(f"Loading cached RNA embeddings from {emb_path}")
        emb = np.load(emb_path)
        if emb.shape[0] == rna_adata.n_obs:
            return emb
        print(f"  Cache shape {emb.shape[0]} != n_cells {rna_adata.n_obs} — re-extracting.")

    try:
        import scgpt
    except ImportError:
        raise ImportError(
            "scgpt not installed. Run: pip install scgpt\n"
            "Or use --rna_model pca for a PCA fallback."
        )

    model_dir = config['scgpt_checkpoint']
    print(f"Extracting scGPT embeddings ({rna_adata.n_obs} cells × "
          f"{rna_adata.n_vars} genes) from {model_dir} ...")

    new_adata = scgpt.tasks.embed_data(
        rna_adata,
        model_dir=model_dir,
        gene_col='index',
        max_length=1200,
        batch_size=64,
        device=device,
        use_fast_transformer=False,
        return_new_adata=True,
    )
    key = 'X_scGPT' if 'X_scGPT' in new_adata.obsm else None
    embeddings = new_adata.obsm[key] if key else new_adata.X
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"  scGPT embeddings: {embeddings.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(emb_path)), exist_ok=True)
    np.save(emb_path, embeddings)
    print(f"  Saved to {emb_path}")
    return embeddings


def extract_rna_embeddings_pca(rna_adata, embed_dim: int = 512) -> np.ndarray:
    """PCA fallback when scGPT is unavailable."""
    import scipy.sparse as sp
    from sklearn.decomposition import PCA

    X = rna_adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)

    n_components = min(embed_dim, X.shape[1] - 1, X.shape[0] - 1)
    print(f"  PCA fallback: {X.shape} → {n_components}d")
    pca = PCA(n_components=n_components, random_state=42)
    Z = pca.fit_transform(X).astype(np.float32)
    if Z.shape[1] < embed_dim:
        pad = np.zeros((Z.shape[0], embed_dim - Z.shape[1]), dtype=np.float32)
        Z = np.concatenate([Z, pad], axis=1)
    return Z


# ============================================================================
# Dataset
# ============================================================================

class PairedBMMCDataset(Dataset):
    """
    In-memory dataset of paired (RNA embedding, ADT array) for BMMC cells.

    Args:
        z_rna:    (N, D_rna) float32 — frozen scGPT embeddings
        adt_mat:  (N, P) float32     — log-normalized ADT counts
        labels:   dict of {label_name: (N,) int array}
        indices:  optional subset (train/val/test)
    """
    def __init__(self, z_rna: np.ndarray, adt_mat: np.ndarray,
                 labels: dict, indices: np.ndarray | None = None):
        self.indices = indices if indices is not None else np.arange(len(z_rna))
        self.z_rna  = torch.from_numpy(z_rna).float()
        self.adt    = torch.from_numpy(adt_mat).float()
        self.labels = {k: torch.from_numpy(np.array(v)) for k, v in labels.items()}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        out = {'z_rna': self.z_rna[idx], 'adt': self.adt[idx]}
        for k, v in self.labels.items():
            out[k] = v[idx]
        return out


def make_donor_splits(obs_df: pd.DataFrame, seed: int = 42,
                      val_donors: int = 1, test_donors: int = 2):
    """
    Hold out entire donors for val/test.
    With 9 donors: 6 train / 1 val / 2 test by default.
    """
    donors = sorted(obs_df['donor'].unique())
    rng    = np.random.RandomState(seed)
    perm   = rng.permutation(donors)
    tr_don = list(perm[:-val_donors - test_donors])
    va_don = list(perm[-val_donors - test_donors:-test_donors])
    te_don = list(perm[-test_donors:])
    print(f"Donor split — train: {tr_don}  val: {va_don}  test: {te_don}")

    tr_idx = np.where(obs_df['donor'].isin(tr_don))[0]
    va_idx = np.where(obs_df['donor'].isin(va_don))[0]
    te_idx = np.where(obs_df['donor'].isin(te_don))[0]
    print(f"  cells — train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}")
    return tr_idx, va_idx, te_idx


# ============================================================================
# Models
# ============================================================================

class ADTEncoder(nn.Module):
    """Trainable MLP encoder for log-normalized ADT surface proteins."""

    def __init__(self, input_dim: int, hidden_dims: list[int],
                 embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        layers, d_in  = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d_in, h), nn.BatchNorm1d(h), nn.ReLU(),
                       nn.Dropout(dropout)]
            d_in = h
        layers.append(nn.Linear(d_in, embed_dim))
        self.net = nn.Sequential(*layers)
        self.embed_dim = embed_dim

    def forward(self, adt: torch.Tensor) -> torch.Tensor:
        return self.net(self.input_bn(adt))


class ProjectionHead(nn.Module):
    """MLP projection head for CA (VICReg / InfoNCE)."""

    def __init__(self, input_dim: int, hidden_dim: int = 512,
                 output_dim: int = 256, num_layers: int = 2):
        super().__init__()
        layers, d_in = [], input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d_in, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU()]
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CrossPredictor(nn.Module):
    """MLP cross-predictor for CP. Bottleneck = last hidden layer."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers, d_in = [], input_dim
        for _ in range(num_layers - 1):
            layers += [nn.Linear(d_in, hidden_dim), nn.LayerNorm(hidden_dim),
                       nn.GELU(), nn.Dropout(dropout)]
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)
        self._bottleneck_end = len(layers) - 1

    def forward(self, x):
        return self.net(x)

    def get_bottleneck(self, x):
        return self.net[:self._bottleneck_end](x)


# ============================================================================
# Losses
# ============================================================================

def vicreg_loss(z1, z2, sim_w=25.0, var_w=25.0, cov_w=1.0):
    sim  = F.mse_loss(z1, z2)
    std1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var  = (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()) / 2
    z1c, z2c = z1 - z1.mean(0), z2 - z2.mean(0)
    N, D = z1c.shape
    c1   = (z1c.T @ z1c) / (N - 1)
    c2   = (z2c.T @ z2c) / (N - 1)
    cov  = (c1.pow(2).sum() - c1.diagonal().pow(2).sum() +
            c2.pow(2).sum() - c2.diagonal().pow(2).sum()) / (2 * D)
    loss = sim_w * sim + var_w * var + cov_w * cov
    return loss, {'sim': sim.item(), 'var': var.item(), 'cov': cov.item(), 'total': loss.item()}


def infonce_loss(z1, z2, temperature=0.1):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(len(z1), device=z1.device)
    loss   = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss, {'infonce': loss.item()}


def cp_loss(pred, target):
    loss    = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=1).mean()
    return loss, {'mse': loss.item(), 'cos_sim': cos_sim.item()}


# ============================================================================
# Training
# ============================================================================

def _get_embeddings(batch, adt_enc, device):
    z_rna = batch['z_rna'].to(device)
    z_adt = adt_enc(batch['adt'].to(device))
    return z_rna, z_adt


def _run_step(mode, z_rna, z_adt, models, ca_loss_type='vicreg', joint_alpha=0.5):
    if mode == 'ca':
        p_a, p_b = models['proj_a'](z_rna), models['proj_b'](z_adt)
        fn = vicreg_loss if ca_loss_type == 'vicreg' else infonce_loss
        return fn(p_a, p_b)

    elif mode == 'cp':
        pred = models['predictor_a2b'](z_rna)
        return cp_loss(pred, z_adt.detach())

    elif mode == 'cp_reverse':
        pred = models['predictor_b2a'](z_adt)
        return cp_loss(pred, z_rna.detach())

    elif mode == 'joint':
        p_a, p_b = models['proj_a'](z_rna), models['proj_b'](z_adt)
        pred_b   = models['predictor_a2b'](z_rna)
        fn = vicreg_loss if ca_loss_type == 'vicreg' else infonce_loss
        l_ca, c_ca = fn(p_a, p_b)
        l_cp, c_cp = cp_loss(pred_b, z_adt.detach())
        loss  = joint_alpha * l_ca + (1 - joint_alpha) * l_cp
        comps = {f'ca_{k}': v for k, v in c_ca.items()}
        comps.update({f'cp_{k}': v for k, v in c_cp.items()})
        comps['total'] = loss.item()
        return loss, comps

    raise ValueError(f"Unknown mode: {mode}")


def train_epoch(mode, loader, models, adt_enc, optimizer, device, ca_loss_type='vicreg'):
    adt_enc.train()
    for m in models.values():
        m.train()
    total, n, comps_acc = 0.0, 0, {}
    for batch in loader:
        z_rna, z_adt = _get_embeddings(batch, adt_enc, device)
        optimizer.zero_grad()
        loss, comps = _run_step(mode, z_rna, z_adt, models, ca_loss_type)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(adt_enc.parameters()) +
            [p for m in models.values() for p in m.parameters()], 1.0)
        optimizer.step()
        total += loss.item()
        for k, v in comps.items():
            comps_acc.setdefault(k, []).append(v)
        n += 1
    avg = {k: float(np.mean(v)) for k, v in comps_acc.items()}
    avg['loss'] = total / max(n, 1)
    return avg


@torch.no_grad()
def validate(mode, loader, models, adt_enc, device, ca_loss_type='vicreg'):
    adt_enc.eval()
    for m in models.values():
        m.eval()
    total, n, comps_acc = 0.0, 0, {}
    for batch in loader:
        z_rna, z_adt = _get_embeddings(batch, adt_enc, device)
        loss, comps  = _run_step(mode, z_rna, z_adt, models, ca_loss_type)
        total += loss.item()
        for k, v in comps.items():
            comps_acc.setdefault(k, []).append(v)
        n += 1
    avg = {k: float(np.mean(v)) for k, v in comps_acc.items()}
    avg['loss'] = total / max(n, 1)
    return avg


@torch.no_grad()
def extract_representations(mode, loader, models, adt_enc, device):
    """Collect all embeddings + labels from a DataLoader (no shuffling)."""
    adt_enc.eval()
    for m in models.values():
        m.eval()

    reps = {'z_rna': [], 'z_adt': []}
    if mode in ('ca', 'joint'):
        reps['proj_rna'], reps['proj_adt'] = [], []
    if mode in ('cp', 'joint'):
        reps['cp_bottleneck'] = []
    if mode == 'cp_reverse':
        reps['cp_rev_bottleneck'] = []

    label_lists = {}
    for batch in loader:
        z_rna, z_adt = _get_embeddings(batch, adt_enc, device)
        reps['z_rna'].append(z_rna.cpu().numpy())
        reps['z_adt'].append(z_adt.cpu().numpy())

        for lbl in SIGNAL_LABELS + NOISE_LABELS:
            if lbl in batch:
                label_lists.setdefault(lbl, []).append(batch[lbl].numpy())

        if mode in ('ca', 'joint') and 'proj_a' in models:
            reps['proj_rna'].append(models['proj_a'](z_rna).cpu().numpy())
            reps['proj_adt'].append(models['proj_b'](z_adt).cpu().numpy())
        if mode in ('cp', 'joint') and 'predictor_a2b' in models:
            reps['cp_bottleneck'].append(
                models['predictor_a2b'].get_bottleneck(z_rna).cpu().numpy())
        if mode == 'cp_reverse' and 'predictor_b2a' in models:
            reps['cp_rev_bottleneck'].append(
                models['predictor_b2a'].get_bottleneck(z_adt).cpu().numpy())

    out = {k: np.concatenate(v, 0) for k, v in reps.items() if v}
    out.update({k: np.concatenate(v, 0) for k, v in label_lists.items()})
    return out


# ============================================================================
# Evaluation
# ============================================================================

def classification_probe(X_train, y_train, X_test, y_test) -> dict:
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_train)
    X_te   = scaler.transform(X_test)
    clf    = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced',
                                solver='lbfgs')
    clf.fit(X_tr, y_train.astype(int))
    y_pred = clf.predict(X_te)
    return {
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'accuracy':          float(accuracy_score(y_test, y_pred)),
        'n_classes':         int(len(np.unique(y_train))),
        'n_train':           int(len(y_train)),
        'n_test':            int(len(y_test)),
    }


def _build_rep_configs(mode: str) -> dict:
    cfg = {
        'frozen_concat': ('z_rna', 'z_adt'),
        'rna_only':      'z_rna',
        'adt_only':      'z_adt',
    }
    if mode in ('ca', 'joint'):
        cfg['ca'] = ('proj_rna', 'proj_adt')
    if mode in ('cp', 'joint'):
        cfg['cp'] = 'cp_bottleneck'
    if mode == 'cp_reverse':
        cfg['cp_reverse'] = 'cp_rev_bottleneck'
    return cfg


def _get_features(reps_tr, reps_te, key):
    if isinstance(key, tuple):
        parts_tr = [reps_tr[k] for k in key if k in reps_tr]
        parts_te = [reps_te[k] for k in key if k in reps_te]
        if not parts_tr:
            return None, None
        return np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1)
    if key not in reps_tr:
        return None, None
    return reps_tr[key], reps_te[key]


def run_evaluation(reps_train: dict, reps_test: dict, mode: str) -> dict:
    """Linear probe on all representation types for signal and noise labels."""
    all_results = {}
    for rep_name, rep_key in _build_rep_configs(mode).items():
        X_tr, X_te = _get_features(reps_train, reps_test, rep_key)
        if X_tr is None:
            continue
        for lbl in SIGNAL_LABELS + NOISE_LABELS:
            if lbl not in reps_train or lbl not in reps_test:
                continue
            y_tr = reps_train[lbl]
            y_te = reps_test[lbl]
            if len(np.unique(y_tr)) < 2:
                continue
            res = classification_probe(X_tr, y_tr, X_te, y_te)
            res['is_signal']      = lbl in SIGNAL_LABELS
            res['interpretation'] = (
                'high = good (signal captured)' if lbl in SIGNAL_LABELS
                else 'high = bad (noise leaking through)'
            )
            all_results[f'{rep_name}/{lbl}'] = res
    return all_results


def print_results(results: dict, mode: str):
    print(f"\n{'='*70}")
    print(f"  Linear Probe Results — mode={mode}")
    print(f"{'='*70}")
    print(f"  {'Representation':<25}  {'Label':<15}  {'BalAcc':>7}  {'Acc':>7}  Note")
    print(f"  {'-'*65}")
    for key, res in sorted(results.items()):
        rep, lbl = key.split('/', 1)
        note = '✓ signal' if res.get('is_signal') else '✗ noise'
        print(f"  {rep:<25}  {lbl:<15}  {res['balanced_accuracy']:>7.3f}"
              f"  {res['accuracy']:>7.3f}  {note}")
    print(f"{'='*70}")


# ============================================================================
# Full training + evaluation pipeline for one mode
# ============================================================================

def build_models(mode: str, rna_dim: int, adt_dim: int,
                 hidden_dim: int, proj_dim: int, proj_layers: int) -> dict:
    models = {}
    if mode in ('ca', 'joint'):
        models['proj_a'] = ProjectionHead(rna_dim, hidden_dim, proj_dim, proj_layers)
        models['proj_b'] = ProjectionHead(adt_dim, hidden_dim, proj_dim, proj_layers)
    if mode in ('cp', 'joint'):
        models['predictor_a2b'] = CrossPredictor(rna_dim, adt_dim, hidden_dim, proj_layers)
    if mode == 'cp_reverse':
        models['predictor_b2a'] = CrossPredictor(adt_dim, rna_dim, hidden_dim, proj_layers)
    return models


def run_mode(mode: str, train_loader, val_loader, test_loader,
             adt_enc: nn.Module, rna_dim: int, adt_dim: int,
             device: torch.device, args: argparse.Namespace,
             output_dir: str) -> dict:
    """Full train → evaluate cycle for one mode."""
    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*60}")

    models = build_models(mode, rna_dim, adt_dim,
                          hidden_dim=args.hidden_dim,
                          proj_dim=args.proj_dim,
                          proj_layers=args.proj_layers)
    for m in models.values():
        m.to(device)

    # Independent ADT encoder copy per mode
    adt_enc_mode = copy.deepcopy(adt_enc).to(device)

    params = list(adt_enc_mode.parameters())
    for m in models.values():
        params += list(m.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val, wait, best_state = float('inf'), 0, None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        tr  = train_epoch(mode, train_loader, models, adt_enc_mode,
                          optimizer, device, args.ca_loss_type)
        val = validate(mode, val_loader, models, adt_enc_mode, device, args.ca_loss_type)
        scheduler.step()
        history['train_loss'].append(tr['loss'])
        history['val_loss'].append(val['loss'])

        if epoch % args.print_every == 0:
            print(f"  Epoch {epoch:4d}/{args.epochs}"
                  f"  train={tr['loss']:.4f}  val={val['loss']:.4f}")

        if val['loss'] < best_val:
            best_val, wait = val['loss'], 0
            best_state = {
                'adt_enc': {k: v.clone() for k, v in adt_enc_mode.state_dict().items()},
                'models':  {n: {k: v.clone() for k, v in m.state_dict().items()}
                            for n, m in models.items()},
            }
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    if best_state is not None:
        adt_enc_mode.load_state_dict(best_state['adt_enc'])
        for n, m in models.items():
            m.load_state_dict(best_state['models'][n])

    # Save checkpoint
    mode_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    torch.save(best_state, os.path.join(mode_dir, 'best_checkpoint.pt'))
    np.save(os.path.join(mode_dir, 'history.npy'), history, allow_pickle=True)

    # Extract representations using non-shuffled loaders
    print("  Extracting representations for linear probing...")
    reps_train = extract_representations(mode, train_loader, models, adt_enc_mode, device)
    reps_test  = extract_representations(mode, test_loader,  models, adt_enc_mode, device)

    results = run_evaluation(reps_train, reps_test, mode)
    print_results(results, mode)

    with open(os.path.join(mode_dir, 'probe_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history['train_loss'], label='train')
    ax.plot(history['val_loss'],   label='val')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.set_title(f'BMMC cross-modal — mode={mode}')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(mode_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    return results


# ============================================================================
# Main
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='BMMC CITE-seq cross-modal CA/CP experiment')
    p.add_argument('--mode', default='all',
                   choices=['ca', 'cp', 'cp_reverse', 'all'],
                   help='Cross-modal training mode (default: all)')
    p.add_argument('--extract_rna_only', action='store_true',
                   help='Only run Stage-0 RNA embedding extraction, then exit')
    p.add_argument('--rna_model', default='scgpt', choices=['scgpt', 'pca'],
                   help='RNA embedding model (default: scgpt)')
    p.add_argument('--epochs',       type=int,   default=200)
    p.add_argument('--batch_size',   type=int,   default=512)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience',     type=int,   default=30)
    p.add_argument('--proj_layers',  type=int,   default=2)
    p.add_argument('--hidden_dim',   type=int,   default=512)
    p.add_argument('--proj_dim',     type=int,   default=256)
    p.add_argument('--ca_loss_type', default='vicreg', choices=['vicreg', 'infonce'])
    p.add_argument('--val_donors',   type=int,   default=1,
                   help='Number of donors held out for validation (default: 1)')
    p.add_argument('--test_donors',  type=int,   default=2,
                   help='Number of donors held out for test (default: 2)')
    p.add_argument('--seed',         type=int,   default=42)
    p.add_argument('--num_workers',  type=int,   default=4)
    p.add_argument('--print_every',  type=int,   default=10)
    p.add_argument('--output_dir',   type=str,   default=None)
    p.add_argument('--adt_pretrained', type=str, default=None,
                   help='Path to pretrained ADTEncoder state dict (.pt). '
                        'If set, backbone is loaded and frozen; only heads are trained.')
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config     = dict(DEFAULT_CONFIG)
    output_dir = args.output_dir or config['output_dir']
    timestamp  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")

    # ---- Load BMMC ---------------------------------------------------------
    sys.path.insert(0, str(MM_ROOT / 'src/bmmc'))
    from bmmc_dataset import load_bmmc

    rna_adata, adt_adata, obs_df = load_bmmc(
        cache_path   = config['h5ad_path'],
        n_top_genes  = config['n_top_genes'],
    )
    import scipy.sparse as sp
    adt_X = adt_adata.X
    if sp.issparse(adt_X):
        adt_X = adt_X.toarray()
    adt_mat = adt_X.astype(np.float32)

    # ---- Stage 0: RNA embeddings -------------------------------------------
    if args.rna_model == 'scgpt':
        rna_embeds = extract_rna_embeddings(
            config, rna_adata,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )
    else:
        rna_embeds = extract_rna_embeddings_pca(rna_adata, config['rna_embed_dim'])
        np.save(config['rna_embeddings'], rna_embeds)

    if args.extract_rna_only:
        print("RNA embeddings extracted. Exiting (--extract_rna_only).")
        return

    # Align: rna_adata and adt_adata share the same cell ordering after load_bmmc
    assert len(rna_embeds) == len(adt_mat) == len(obs_df), (
        f"Alignment error: rna={len(rna_embeds)}, adt={len(adt_mat)}, obs={len(obs_df)}"
    )

    # ---- Encode labels to integers -----------------------------------------
    from sklearn.preprocessing import LabelEncoder
    ct_enc  = LabelEncoder().fit(obs_df['cell_type'])
    don_enc = LabelEncoder().fit(obs_df['DonorNumber'].astype(str))
    labels  = {
        'cell_type': ct_enc.transform(obs_df['cell_type']).astype(np.int64),
        'donor':     don_enc.transform(obs_df['DonorNumber'].astype(str)).astype(np.int64),
    }
    print(f"\nCell types ({len(ct_enc.classes_)}): {list(ct_enc.classes_)[:6]} ...")
    print(f"Donors ({len(don_enc.classes_)}): {list(don_enc.classes_)}")

    # ---- Splits (donor-based hold-out) -------------------------------------
    # Use obs_df['DonorNumber'] for splitting, stored as 'donor' in labels
    obs_df_split = obs_df.copy()
    obs_df_split['donor'] = obs_df['DonorNumber'].astype(str)
    tr_idx, va_idx, te_idx = make_donor_splits(
        obs_df_split, seed=args.seed,
        val_donors=args.val_donors, test_donors=args.test_donors,
    )

    def make_loader(idx, shuffle):
        ds = PairedBMMCDataset(rna_embeds, adt_mat, labels, indices=idx)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=shuffle)

    # Non-shuffled train loader for representation extraction
    train_loader      = make_loader(tr_idx, shuffle=True)
    train_loader_eval = make_loader(tr_idx, shuffle=False)
    val_loader        = make_loader(va_idx, shuffle=False)
    test_loader       = make_loader(te_idx, shuffle=False)

    rna_dim = rna_embeds.shape[1]
    adt_dim = config['adt_embed_dim']
    print(f"\nData: train={len(tr_idx)}, val={len(va_idx)}, test={len(te_idx)}"
          f", n_proteins={adt_mat.shape[1]}, rna_dim={rna_dim}, adt_dim={adt_dim}")

    # ---- ADT Encoder (shared init, deep-copied per mode) -------------------
    adt_enc = ADTEncoder(
        input_dim   = adt_mat.shape[1],
        hidden_dims = config['adt_hidden_dims'],
        embed_dim   = config['adt_embed_dim'],
        dropout     = 0.1,
    )
    if args.adt_pretrained:
        ckpt = torch.load(args.adt_pretrained, map_location='cpu')
        adt_enc.load_state_dict(ckpt)
        print(f"ADTEncoder: loaded pretrained weights from {args.adt_pretrained} (fine-tuning from pretrained init)")
    n_params = sum(p.numel() for p in adt_enc.parameters() if p.requires_grad)
    print(f"ADTEncoder trainable params: {n_params:,}")

    # ---- Save config -------------------------------------------------------
    run_config = {
        'config':     {k: v for k, v in config.items()},
        'args':       vars(args),
        'n_cells':    int(len(rna_embeds)),
        'rna_dim':    int(rna_dim),
        'adt_dim':    int(adt_dim),
        'n_proteins': int(adt_mat.shape[1]),
        'cell_types': list(ct_enc.classes_),
        'donors':     list(don_enc.classes_),
        'proteins':   adt_adata.var_names.tolist(),
        'train_donors': list(obs_df_split.iloc[tr_idx]['donor'].unique()),
        'val_donors':   list(obs_df_split.iloc[va_idx]['donor'].unique()),
        'test_donors':  list(obs_df_split.iloc[te_idx]['donor'].unique()),
    }
    with open(os.path.join(output_dir, 'run_config.json'), 'w') as f:
        json.dump(run_config, f, indent=2)

    # ---- Run modes ---------------------------------------------------------
    modes = ['ca', 'cp', 'cp_reverse'] if args.mode == 'all' else [args.mode]
    all_results = {}

    for mode in modes:
        try:
            # Swap shuffled train_loader for training, non-shuffled for extraction
            results = _run_mode_with_eval_loader(
                mode, train_loader, train_loader_eval, val_loader, test_loader,
                adt_enc, rna_dim, adt_dim, device, args, output_dir,
            )
            all_results[mode] = results
        except Exception as e:
            import traceback
            print(f"Mode {mode} failed: {e}")
            traceback.print_exc()

    # ---- Summary -----------------------------------------------------------
    print(f"\n{'='*70}")
    print("  SUMMARY — balanced_accuracy by mode")
    print(f"{'='*70}")
    print(f"  {'Representation':<25}  {'CA':>7}  {'CP':>7}  {'CP_rev':>7}")
    for rep in ['rna_only', 'adt_only', 'frozen_concat', 'ca', 'cp', 'cp_reverse']:
        row = f"  {rep:<25}"
        for m in ['ca', 'cp', 'cp_reverse']:
            key = f'{rep}/cell_type'
            v   = all_results.get(m, {}).get(key, {}).get('balanced_accuracy', float('nan'))
            row += f"  {v:>7.3f}"
        print(row)
    print(f"{'='*70}")

    with open(os.path.join(output_dir, 'all_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_dir}")


def _run_mode_with_eval_loader(
    mode, train_loader, train_loader_eval, val_loader, test_loader,
    adt_enc, rna_dim, adt_dim, device, args, output_dir,
):
    """Like run_mode but uses a separate non-shuffled train loader for extraction."""
    print(f"\n{'='*60}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*60}")

    models = build_models(mode, rna_dim, adt_dim,
                          hidden_dim=args.hidden_dim,
                          proj_dim=args.proj_dim,
                          proj_layers=args.proj_layers)
    for m in models.values():
        m.to(device)

    adt_enc_mode = copy.deepcopy(adt_enc).to(device)

    params = list(adt_enc_mode.parameters())
    for m in models.values():
        params += list(m.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val, wait, best_state = float('inf'), 0, None
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(1, args.epochs + 1):
        tr  = train_epoch(mode, train_loader, models, adt_enc_mode,
                          optimizer, device, args.ca_loss_type)
        val = validate(mode, val_loader, models, adt_enc_mode, device, args.ca_loss_type)
        scheduler.step()
        history['train_loss'].append(tr['loss'])
        history['val_loss'].append(val['loss'])

        if epoch % args.print_every == 0:
            print(f"  Epoch {epoch:4d}/{args.epochs}"
                  f"  train={tr['loss']:.4f}  val={val['loss']:.4f}")

        if val['loss'] < best_val:
            best_val, wait = val['loss'], 0
            best_state = {
                'adt_enc': {k: v.clone() for k, v in adt_enc_mode.state_dict().items()},
                'models':  {n: {k: v.clone() for k, v in m.state_dict().items()}
                            for n, m in models.items()},
            }
        else:
            wait += 1
            if wait >= args.patience:
                print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
                break

    if best_state is not None:
        adt_enc_mode.load_state_dict(best_state['adt_enc'])
        for n, m in models.items():
            m.load_state_dict(best_state['models'][n])

    mode_dir = os.path.join(output_dir, mode)
    os.makedirs(mode_dir, exist_ok=True)
    torch.save(best_state, os.path.join(mode_dir, 'best_checkpoint.pt'))
    np.save(os.path.join(mode_dir, 'history.npy'), history, allow_pickle=True)

    print("  Extracting representations for linear probing...")
    reps_train = extract_representations(mode, train_loader_eval, models, adt_enc_mode, device)
    reps_test  = extract_representations(mode, test_loader,       models, adt_enc_mode, device)

    results = run_evaluation(reps_train, reps_test, mode)
    print_results(results, mode)

    with open(os.path.join(mode_dir, 'probe_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history['train_loss'], label='train')
    ax.plot(history['val_loss'],   label='val')
    ax.set_xlabel('epoch'); ax.set_ylabel('loss')
    ax.set_title(f'BMMC cross-modal — mode={mode}')
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(mode_dir, 'loss_curve.png'), dpi=150)
    plt.close(fig)

    return results


if __name__ == '__main__':
    main()
