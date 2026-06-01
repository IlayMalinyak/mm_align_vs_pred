"""
Cross-Modal CA vs CP Experiment: LAMOST (spectra) x TESS (light curves)

Tests phase diagram prediction from the spiked covariance model framework.
Phase estimation for LAMOST x TESS:
    kappa=0.090, nu=0.931, Delta_CA=0.069 (<1), Delta_CP=0.138 (<1) -> "Neither"
Theory predicts neither method recovers shared signal well.

Two data modes:
  --use_cached_features : train on pre-extracted feature arrays (fast)
  (default)             : load raw data on-the-fly through frozen pretrained encoders

Three training modes:
  - CA (Cross-Alignment): projection heads + VICReg loss
  - CP (Cross-Prediction): MLP predictor + MSE loss
  - Joint: both losses weighted

Evaluation: linear probe (Ridge regression) on frozen representations.

Usage:
    python src/cross_modal_tess.py --mode all --use_cached_features
    python src/cross_modal_tess.py --mode ca
"""

from __future__ import annotations
import os
import sys
import argparse
import json
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Configuration
# ============================================================================

MULTIDESA_ROOT = Path(os.environ.get('MULTIDESA_ROOT', str(Path(__file__).parent)))

DEFAULT_CONFIG = {
    # Raw data paths (on-the-fly mode)
    'crossmatch_parquet': os.environ.get('CROSSMATCH_PARQUET', '/path/to/multimodal_df_min2.parquet'),
    'lamost_hdf5_dir': os.environ.get('LAMOST_HDF5_DIR', '/path/to/lamost/hdf5_data'),
    # Pretrained encoder checkpoints
    'lamost_checkpoint': str(MULTIDESA_ROOT / 'logs/lamost/2025-11-02/23018/lamost_SpectralViT.pth'),
    'tess_checkpoint': str(MULTIDESA_ROOT / 'logs/tess/2026-03-13-10-46/tess_sap_all_DoubleInputRegressor.pth'),
    # Config files (for model architecture params)
    'lamost_config': str(MULTIDESA_ROOT / 'configs/lamost.yaml'),
    'tess_config': str(MULTIDESA_ROOT / 'configs/tess.yaml'),
    # Output
    'output_dir': str(MULTIDESA_ROOT / 'logs/cross_modal/lamost_x_tess'),
}

EVAL_LABELS = ['teff', 'logg', 'feh', 'prot']

# Shared-signal evaluation catalogs (matched via gaia_source_id -> KID -> catalog)
AGE_CATALOG = os.environ.get('AGE_CATALOG', '/path/to/ages_dataset_gyro.csv')
NSS_CATALOG = os.environ.get('NSS_CATALOG', '/path/to/nss_dataset.csv')

PHASE_PARAMS = {
    'kappa': 0.090, 'nu': 0.931,
    'gamma_x': 0.164, 'gamma_y': 0.085,
    'gamma_tilde_x': 0.003, 'gamma_tilde_y': 0.023,
    'Delta_CA': 0.069, 'Delta_CP': 0.138,
    'predicted_regime': 'Neither',
    'n_matched': 153,
}


# ============================================================================
# On-the-fly raw data loading
# ============================================================================

class CrossModalPairDataset(Dataset):
    """Load raw LAMOST spectra (HDF5) + TESS light curves (HDF5) on-the-fly."""

    def __init__(self, df: pd.DataFrame, lamost_hdf5_dir: str,
                 lamost_transforms=None, tess_transforms=None,
                 lamost_seq_len: int = 4096, tess_seq_len: int = 14000):
        self.df = df.reset_index(drop=True)
        self.lamost_hdf5_dir = lamost_hdf5_dir
        self.lamost_transforms = lamost_transforms
        self.tess_transforms = tess_transforms
        self.lamost_seq_len = lamost_seq_len
        self.tess_seq_len = tess_seq_len

        # HDF5 caches (per-worker, populated lazily)
        self._h5_handles = {}
        self._h5_index = {}

    def __len__(self):
        return len(self.df)

    # -- HEALPix lookup ---------------------------------------------------

    @staticmethod
    def _healpix_index(ra, dec, nside=16):
        from astropy_healpix import HEALPix
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        hp = HEALPix(nside=nside, order='ring', frame='icrs')
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')
        return int(hp.skycoord_to_healpix(coord))

    def _get_h5(self, hp_idx):
        """Open (and cache) the HDF5 file for a HEALPix tile."""
        import h5py
        path = os.path.join(self.lamost_hdf5_dir, f'healpix_{hp_idx}.h5')
        if path not in self._h5_handles:
            if len(self._h5_handles) >= 8:
                old = next(iter(self._h5_handles))
                self._h5_handles[old].close()
                del self._h5_handles[old]
                del self._h5_index[old]
            self._h5_handles[path] = h5py.File(path, 'r')
            desigs = self._h5_handles[path]['designation'][:]
            idx_map = {}
            for i, d in enumerate(desigs):
                d_str = d.decode('utf-8') if isinstance(d, bytes) else str(d)
                if d_str.endswith('.0'):
                    d_str = d_str[:-2]
                idx_map[d_str] = i
            self._h5_index[path] = idx_map
        return self._h5_handles[path], self._h5_index[path]

    def _load_lamost(self, row):
        """Load a single LAMOST spectrum from HDF5."""
        obsid = int(row['lamost_obsid'])
        ra, dec = row['ra'], row['dec']
        hp_idx = self._healpix_index(ra, dec)
        hf, idx_map = self._get_h5(hp_idx)

        obsid_str = str(obsid)
        if obsid_str not in idx_map:
            raise KeyError(f"LAMOST obsid {obsid_str} not in healpix_{hp_idx}.h5")

        i = idx_map[obsid_str]
        flux = hf['flux'][i].astype(np.float32)
        wv = hf['wavelength'][i].astype(np.float32)

        info = {'wavelength': wv, 'RV': 0.0}
        if self.lamost_transforms is not None:
            flux, _, info = self.lamost_transforms(flux, None, info)

        flux = self._pad_or_trim(flux, self.lamost_seq_len)
        return flux

    def _load_tess(self, row):
        """Load a single TESS light curve from HDF5 via tess_paths."""
        import h5py

        paths_str = row.get('tess_paths', '[]')
        if isinstance(paths_str, float) and np.isnan(paths_str):
            raise KeyError("No TESS paths")
        paths = json.loads(paths_str) if isinstance(paths_str, str) else paths_str
        if not paths:
            raise KeyError("Empty TESS paths")

        # Pick the longest-available sector (by idx count)
        path_dict = paths[0]  # default: first sector
        if len(paths) > 1:
            # Pick a random sector for variability
            path_dict = paths[np.random.randint(0, len(paths))]

        with h5py.File(path_dict['path'], 'r') as f:
            flux_key = 'sap_flux' if 'sap_flux' in f else 'flux'
            x = f[flux_key][path_dict['idx']].astype(np.float32)
            time_arr = f['time'][path_dict['idx']].astype(np.float32)

        # Remove NaNs
        valid = np.isfinite(x) & np.isfinite(time_arr)
        x = x[valid]
        time_arr = time_arr[valid]
        if len(time_arr) > 0:
            time_arr = time_arr - time_arr[0]

        info = {
            'time': time_arr,
            'cadence_min': 2.0,  # TESS 2-min cadence
        }

        if self.tess_transforms is not None:
            x, _, info = self.tess_transforms(x, None, info)

        # Stack channels: flux + ACF + FFT/LombScargle
        channels = [self._pad_or_trim(x, self.tess_seq_len)]
        for key in ['acf', 'fft', 'ls_power']:
            if key in info and info[key] is not None:
                ch = info[key]
                if isinstance(ch, np.ndarray):
                    if ch.ndim == 2:
                        ch = ch[0]
                    channels.append(self._pad_or_trim(ch, self.tess_seq_len))
        return np.stack(channels, axis=0)  # (C, L)

    @staticmethod
    def _pad_or_trim(x, target_len):
        """Pad or trim 1D array to target length."""
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = np.asarray(x, dtype=np.float32)
        if x.ndim > 1:
            x = x.ravel()
        if len(x) >= target_len:
            return x[:target_len]
        return np.pad(x, (0, target_len - len(x)), mode='constant')

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            flux_lamost = self._load_lamost(row)
            flux_tess = self._load_tess(row)
        except Exception as e:
            print(f"[WARN] skip idx={idx} TIC={row.get('TIC')}: {e}")
            flux_lamost = np.zeros(self.lamost_seq_len, dtype=np.float32)
            flux_tess = np.zeros((4, self.tess_seq_len), dtype=np.float32)

        labels = np.array([row.get(c, np.nan) for c in EVAL_LABELS], dtype=np.float32)

        return {
            'flux_lamost': torch.from_numpy(flux_lamost),
            'flux_tess': torch.from_numpy(flux_tess),
            'labels': torch.from_numpy(labels),
        }


# ============================================================================
# Frozen encoder wrappers
# ============================================================================

def _load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class _Container:
    """Lightweight attribute container (mirrors MultiDESA's util.utils.Container)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


def build_lamost_encoder(config: dict, device: torch.device) -> nn.Module:
    """Instantiate frozen SpectralViT from config + checkpoint."""
    sys.path.insert(0, str(MULTIDESA_ROOT))
    from nn.spectra_models import SpectralViT
    from nn.utils import load_checkpoints_ddp

    cfg = _load_yaml(config['lamost_config'])
    model_args = _Container(**cfg['SpectralViT'])
    conformer_args = _Container(**cfg['Conformer'])

    model = SpectralViT(model_args, conformer_args)
    model = load_checkpoints_ddp(model, config['lamost_checkpoint'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  LAMOST encoder: SpectralViT, cls_dim={conformer_args.encoder_dim}")
    return model


def build_tess_encoder(config: dict, device: torch.device) -> nn.Module:
    """Instantiate frozen DoubleInputRegressor from TESS config + checkpoint."""
    sys.path.insert(0, str(MULTIDESA_ROOT))
    from nn.models import CNNEncoder, DoubleInputRegressor
    from nn.astroconf import Astroconformer
    from nn.utils import init_model
    from nn.Modules.mhsa_pro import RotaryEmbedding
    from nn.Modules.conformer import ConformerEncoder

    cfg = _load_yaml(config['tess_config'])
    model_args = _Container(**cfg['DoubleInputRegressor'])
    cnn_args = _Container(**cfg['CNNEncoder'])
    astroconf_args = _Container(**cfg['AstroConformer'])
    conformer_args = _Container(**cfg['Conformer'])

    # Build sub-encoders
    encoder1 = CNNEncoder(cnn_args)
    encoder2 = Astroconformer(astroconf_args)
    encoder2 = init_model(encoder2, astroconf_args)

    # Mixer (Conformer) + rotary embeddings
    head_size = conformer_args.encoder_dim // conformer_args.num_heads
    rotary_ndims = int(head_size * 0.5)
    pe = RotaryEmbedding(rotary_ndims)
    mixer = ConformerEncoder(conformer_args)

    model = DoubleInputRegressor(encoder1, encoder2, model_args, mixer=mixer, rope=pe)

    # Load checkpoint with key prefix stripping
    from collections import OrderedDict
    raw = torch.load(config['tess_checkpoint'], map_location='cpu')
    cleaned = OrderedDict()
    for k, v in raw.items():
        while k.startswith('module.'):
            k = k[7:]
        if k.startswith('model.backbone.'):
            k = k[len('model.backbone.'):]
        else:
            continue  # skip wrapper keys (model.projector, model.predictor, etc.)
        cleaned[k] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    print(f"  TESS ckpt: loaded {len(cleaned)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    print(f"  TESS encoder: DoubleInputRegressor, feature_dim={model_args.feature_dim}")
    return model


@torch.no_grad()
def extract_features_batch(lamost_enc, tess_enc, batch, device):
    """Run frozen encoders on a raw-data batch -> (z_a, z_b)."""
    flux_l = batch['flux_lamost'].to(device)  # (B, L)
    flux_t = batch['flux_tess'].to(device)    # (B, C, L)

    # LAMOST -> CLS token
    out_l = lamost_enc(flux_l, return_all=True)
    z_a = out_l['cls']  # (B, dim_lamost)

    # TESS -> pooled encoding
    out_t = tess_enc(flux_t, return_all=True)
    z_b = out_t['tokens']  # (B, dim_tess) or (B, seq, dim)
    if z_b.dim() == 3:
        z_b = z_b.mean(1)

    return z_a, z_b


# ============================================================================
# Cached feature loading
# ============================================================================

class PairedFeatureDataset(Dataset):
    """Dataset of paired (LAMOST, TESS) pre-extracted features."""
    def __init__(self, z_a, z_b, labels=None):
        self.z_a = torch.from_numpy(z_a).float()
        self.z_b = torch.from_numpy(z_b).float()
        self.labels = torch.from_numpy(labels).float() if labels is not None else None

    def __len__(self):
        return len(self.z_a)

    def __getitem__(self, idx):
        out = {'z_a': self.z_a[idx], 'z_b': self.z_b[idx]}
        if self.labels is not None:
            out['labels'] = self.labels[idx]
        return out


def load_cached_splits(cache_dir):
    """Load pre-extracted features from extract_cross_modal_features_tess.py output."""
    splits = {}
    for name in ['train', 'val', 'test']:
        z_l = np.load(os.path.join(cache_dir, f'z_lamost_{name}.npy'))
        z_t = np.load(os.path.join(cache_dir, f'z_tess_{name}.npy'))
        meta = pd.read_parquet(os.path.join(cache_dir, f'meta_{name}.parquet'))
        label_cols = [c for c in EVAL_LABELS if c in meta.columns]
        # Metadata columns for shared-signal evaluation
        meta_cols = ['gaia_source_id', 'TIC', 'KID', 'lamost_obsid', 'ra', 'dec']
        meta_cols = [c for c in meta_cols if c in meta.columns]
        splits[name] = {
            'z_lamost': z_l,
            'z_tess': z_t,
            'labels': meta[label_cols].reset_index(drop=True),
            'metadata': meta[meta_cols].reset_index(drop=True),
        }
        print(f"  {name}: {len(z_l)} samples, LAMOST={z_l.shape[1]}d, TESS={z_t.shape[1]}d")
    return splits['train'], splits['val'], splits['test']


# ============================================================================
# Models (CA/CP heads) — identical to cross_modal.py
# ============================================================================

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=256,
                 num_layers=2, use_bn=True):
        super().__init__()
        layers = []
        d_in = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CrossPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512,
                 num_layers=2, dropout=0.1):
        super().__init__()
        layers = []
        d_in = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d_in, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, output_dim))
        self.net = nn.Sequential(*layers)
        self._bottleneck_end = len(layers) - 1

    def forward(self, x):
        return self.net(x)

    def get_bottleneck(self, x):
        """Return the bottleneck (hidden) representation before the final linear."""
        return self.net[:self._bottleneck_end](x)


# ============================================================================
# Losses
# ============================================================================

def vicreg_loss(z1, z2, sim_weight=25.0, var_weight=25.0, cov_weight=1.0):
    sim_loss = F.mse_loss(z1, z2)
    std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
    std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
    var_loss = (F.relu(1.0 - std_z1).mean() + F.relu(1.0 - std_z2).mean()) / 2
    z1_c, z2_c = z1 - z1.mean(0), z2 - z2.mean(0)
    N, D = z1_c.shape
    cov1 = (z1_c.T @ z1_c) / (N - 1)
    cov2 = (z2_c.T @ z2_c) / (N - 1)
    off1 = cov1.pow(2).sum() - cov1.diagonal().pow(2).sum()
    off2 = cov2.pow(2).sum() - cov2.diagonal().pow(2).sum()
    cov_loss = (off1 + off2) / (2 * D)
    total = sim_weight * sim_loss + var_weight * var_loss + cov_weight * cov_loss
    return total, {'sim': sim_loss.item(), 'var': var_loss.item(),
                   'cov': cov_loss.item(), 'total': total.item()}


def infonce_loss(z1, z2, temperature=0.1):
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(len(z1), device=z1.device)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
    return loss, {'infonce': loss.item()}


def cp_loss(pred, target):
    loss = F.mse_loss(pred, target)
    cos_sim = F.cosine_similarity(pred, target, dim=1).mean()
    return loss, {'mse': loss.item(), 'cos_sim': cos_sim.item()}


# ============================================================================
# Training loop
# ============================================================================

def _run_step(mode, z_a, z_b, models, ca_loss_type='vicreg', joint_alpha=0.5):
    """Forward pass + loss for a single batch."""
    if mode == 'ca':
        p_a, p_b = models['proj_a'](z_a), models['proj_b'](z_b)
        loss, comps = (vicreg_loss if ca_loss_type == 'vicreg' else infonce_loss)(p_a, p_b)
    elif mode == 'cp':
        pred_b = models['predictor_a2b'](z_a)
        loss, comps = cp_loss(pred_b, z_b.detach())
    elif mode == 'cp_reverse':
        pred_a = models['predictor_b2a'](z_b)
        loss, comps = cp_loss(pred_a, z_a.detach())
    elif mode == 'joint':
        p_a, p_b = models['proj_a'](z_a), models['proj_b'](z_b)
        pred_b = models['predictor_a2b'](z_a)
        l_ca, c_ca = (vicreg_loss if ca_loss_type == 'vicreg' else infonce_loss)(p_a, p_b)
        l_cp, c_cp = cp_loss(pred_b, z_b.detach())
        loss = joint_alpha * l_ca + (1 - joint_alpha) * l_cp
        comps = {f'ca_{k}': v for k, v in c_ca.items()}
        comps.update({f'cp_{k}': v for k, v in c_cp.items()})
        comps['total'] = loss.item()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return loss, comps


def train_epoch(mode, loader, models, optimizer, device,
                ca_loss_type='vicreg', joint_alpha=0.5,
                lamost_enc=None, tess_enc=None):
    for m in models.values():
        m.train()
    total_loss, n = 0, 0
    all_comps = {}

    for batch in loader:
        if lamost_enc is not None:
            z_a, z_b = extract_features_batch(lamost_enc, tess_enc, batch, device)
        else:
            z_a, z_b = batch['z_a'].to(device), batch['z_b'].to(device)

        optimizer.zero_grad()
        loss, comps = _run_step(mode, z_a, z_b, models, ca_loss_type, joint_alpha)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for m in models.values() for p in m.parameters()], 1.0)
        optimizer.step()

        total_loss += loss.item()
        for k, v in comps.items():
            all_comps.setdefault(k, []).append(v)
        n += 1

    avg = {k: np.mean(v) for k, v in all_comps.items()}
    avg['loss'] = total_loss / max(n, 1)
    return avg


@torch.no_grad()
def validate(mode, loader, models, device, ca_loss_type='vicreg', joint_alpha=0.5,
             lamost_enc=None, tess_enc=None):
    for m in models.values():
        m.eval()
    total_loss, n = 0, 0
    all_comps = {}

    for batch in loader:
        if lamost_enc is not None:
            z_a, z_b = extract_features_batch(lamost_enc, tess_enc, batch, device)
        else:
            z_a, z_b = batch['z_a'].to(device), batch['z_b'].to(device)

        loss, comps = _run_step(mode, z_a, z_b, models, ca_loss_type, joint_alpha)
        total_loss += loss.item()
        for k, v in comps.items():
            all_comps.setdefault(k, []).append(v)
        n += 1

    avg = {k: np.mean(v) for k, v in all_comps.items()}
    avg['loss'] = total_loss / max(n, 1)
    return avg


@torch.no_grad()
def extract_representations(mode, loader, models, device,
                            lamost_enc=None, tess_enc=None):
    """Extract learned representations for linear probing."""
    for m in models.values():
        m.eval()

    reps = {'z_a': [], 'z_b': [], 'labels': []}
    if mode in ('ca', 'joint'):
        reps['proj_a'], reps['proj_b'] = [], []
    if mode in ('cp', 'joint'):
        reps['cp_bottleneck'] = []
    if mode == 'cp_reverse':
        reps['cp_rev_bottleneck'] = []

    for batch in loader:
        if lamost_enc is not None:
            z_a, z_b = extract_features_batch(lamost_enc, tess_enc, batch, device)
        else:
            z_a, z_b = batch['z_a'].to(device), batch['z_b'].to(device)

        reps['z_a'].append(z_a.cpu().numpy())
        reps['z_b'].append(z_b.cpu().numpy())
        if 'labels' in batch:
            reps['labels'].append(batch['labels'].numpy())

        if mode in ('ca', 'joint') and 'proj_a' in models:
            reps['proj_a'].append(models['proj_a'](z_a).cpu().numpy())
            reps['proj_b'].append(models['proj_b'](z_b).cpu().numpy())
        if mode in ('cp', 'joint') and 'predictor_a2b' in models:
            reps['cp_bottleneck'].append(
                models['predictor_a2b'].get_bottleneck(z_a).cpu().numpy())
        if mode == 'cp_reverse' and 'predictor_b2a' in models:
            reps['cp_rev_bottleneck'].append(
                models['predictor_b2a'].get_bottleneck(z_b).cpu().numpy())

    return {k: np.concatenate(v, axis=0) for k, v in reps.items() if v}


# ============================================================================
# Evaluation: Linear Probe
# ============================================================================

def linear_probe(train_feats, train_labels, test_feats, test_labels, alpha=1.0):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_feats)
    X_test = scaler.transform(test_feats)
    results = {}
    if train_labels.ndim == 1:
        train_labels, test_labels = train_labels[:, None], test_labels[:, None]
    for j in range(train_labels.shape[1]):
        mask_tr = ~np.isnan(train_labels[:, j])
        mask_te = ~np.isnan(test_labels[:, j])
        if mask_tr.sum() < 10 or mask_te.sum() < 5:
            continue
        model = Ridge(alpha=alpha)
        model.fit(X_train[mask_tr], train_labels[mask_tr, j])
        y_pred = model.predict(X_test[mask_te])
        results[j] = {
            'r2': r2_score(test_labels[mask_te, j], y_pred),
            'rmse': float(np.sqrt(mean_squared_error(test_labels[mask_te, j], y_pred))),
            'n_test': int(mask_te.sum()),
        }
    return results


def _build_rep_configs(mode):
    """Build representation configs: frozen baseline + unified method features."""
    rep_configs = {
        'frozen_concat': ('z_a', 'z_b'),
        'lamost_only': 'z_a',
        'tess_only': 'z_b',
    }
    if mode in ('ca', 'joint'):
        rep_configs['ca'] = ('proj_a', 'proj_b')
    if mode in ('cp', 'joint'):
        rep_configs['cp'] = 'cp_bottleneck'
    if mode == 'cp_reverse':
        rep_configs['cp_reverse'] = 'cp_rev_bottleneck'
    return rep_configs


def _get_features(reps_train, reps_test, rep_key):
    """Extract feature matrices from reps dicts for a given rep_key."""
    if isinstance(rep_key, tuple):
        parts_tr = [reps_train[k] for k in rep_key if k in reps_train]
        parts_te = [reps_test[k] for k in rep_key if k in reps_test]
        if not parts_tr:
            return None, None
        return np.concatenate(parts_tr, 1), np.concatenate(parts_te, 1)
    else:
        if rep_key not in reps_train:
            return None, None
        return reps_train[rep_key], reps_test[rep_key]


def run_evaluation(reps_train, reps_test, label_names, mode):
    y_train, y_test = reps_train.get('labels'), reps_test.get('labels')
    if y_train is None or y_test is None:
        return {}

    rep_configs = _build_rep_configs(mode)

    all_results = {}
    for rep_name, rep_key in rep_configs.items():
        X_tr, X_te = _get_features(reps_train, reps_test, rep_key)
        if X_tr is None:
            continue

        for j, res in linear_probe(X_tr, y_train, X_te, y_test).items():
            lbl = label_names[j] if j < len(label_names) else f'label_{j}'
            all_results[f'{rep_name}/{lbl}'] = res

    # Supervised ceiling: MLP on frozen concat for each label
    X_tr_cat, X_te_cat = _get_features(reps_train, reps_test, ('z_a', 'z_b'))
    if X_tr_cat is not None:
        if y_train.ndim == 1:
            y_train_2d, y_test_2d = y_train[:, None], y_test[:, None]
        else:
            y_train_2d, y_test_2d = y_train, y_test
        for j in range(y_train_2d.shape[1]):
            mask_tr = ~np.isnan(y_train_2d[:, j])
            mask_te = ~np.isnan(y_test_2d[:, j])
            if mask_tr.sum() < 50 or mask_te.sum() < 20:
                continue
            lbl = label_names[j] if j < len(label_names) else f'label_{j}'
            print(f"  Computing supervised ceiling ({lbl})...")
            res = supervised_ceiling_regression(
                X_tr_cat[mask_tr], y_train_2d[mask_tr, j],
                X_te_cat[mask_te], y_test_2d[mask_te, j])
            res['n_test'] = int(mask_te.sum())
            all_results[f'supervised_ceiling/{lbl}'] = res

    return all_results


# ============================================================================
# Shared-signal evaluation: Age (regression) + Binarity (classification)
# Uses gaia_source_id to match with KID-based catalogs via the multimodal catalog
# ============================================================================

def _build_gaia_to_kid_map():
    """Build gaia_source_id -> KID mapping from the multimodal catalog."""
    df = pd.read_parquet(DEFAULT_CONFIG['crossmatch_parquet'],
                         columns=['gaia_source_id', 'KID'])
    # Keep only rows with valid KID
    df = df.dropna(subset=['KID'])
    df['KID'] = df['KID'].astype(int)
    # Deduplicate (one gaia -> one KID)
    df = df.drop_duplicates('gaia_source_id')
    return dict(zip(df['gaia_source_id'].values, df['KID'].values))


def _load_age_labels(kids):
    """Load gyro age labels for given KIDs. Returns array of ages (NaN where missing)."""
    age_df = pd.read_csv(AGE_CATALOG, low_memory=False)
    age_df = age_df.dropna(subset=['final_age', 'age_error'])
    age_map = dict(zip(age_df['KID'].values, age_df['final_age'].values))
    return np.array([age_map.get(k, np.nan) for k in kids], dtype=np.float32)


def _load_binarity_labels(kids):
    """Load binarity labels for given KIDs. Returns array of 0/1 (NaN where missing).
    Following finetune_nss.py: exclude class 3, remap 4->3, then hard binary = class > 0."""
    nss_df = pd.read_csv(NSS_CATALOG, low_memory=False)
    nss_df = nss_df[nss_df['binarity_class'] != 3]
    nss_df.loc[nss_df['binarity_class'] == 4, 'binarity_class'] = 3
    nss_df['binary_hard'] = (nss_df['binarity_class'] > 0).astype(int)
    nss_map = dict(zip(nss_df['KID'].values, nss_df['binary_hard'].values))
    return np.array([nss_map.get(k, np.nan) for k in kids], dtype=np.float32)


def classification_probe(X_train, y_train, X_test, y_test):
    """Logistic regression classification probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    model = LogisticRegression(max_iter=5000, C=1.0, class_weight='balanced')
    model.fit(X_tr, y_train.astype(int))
    y_pred = model.predict(X_te)

    return {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_pred)),
        'f1': float(f1_score(y_test, y_pred, average='binary')),
        'n_train': len(y_train),
        'n_test': len(y_test),
    }


def supervised_ceiling_regression(X_train, y_train, X_test, y_test,
                                   hidden_dim=256, epochs=200, lr=1e-3,
                                   patience=20, batch_size=256):
    """Train a small MLP directly on the downstream label. Returns R², RMSE."""
    from torch.utils.data import TensorDataset, DataLoader as TDL

    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_te_np = y_test

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, 1),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = TDL(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_loss, wait, best_state = float('inf'), 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.mse_loss(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_pred = model(X_te.to(device)).cpu().numpy().ravel()
        val_loss = float(np.mean((val_pred - y_te_np) ** 2))
        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = model(X_te.to(device)).cpu().numpy().ravel()
    return {
        'r2': float(r2_score(y_te_np, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_te_np, y_pred))),
    }


def supervised_ceiling_classification(X_train, y_train, X_test, y_test,
                                      hidden_dim=256, epochs=200, lr=1e-3,
                                      patience=20, batch_size=256):
    """Train a small MLP directly on a binary label. Returns accuracy, balanced_accuracy, f1."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from torch.utils.data import TensorDataset, DataLoader as TDL

    scaler = StandardScaler()
    X_tr = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
    X_te = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_te_np = y_test.astype(int)

    # Class weighting
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        nn.Linear(X_tr.shape[1], hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(hidden_dim, 1),
    ).to(device)
    pos_weight = pos_weight.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = TDL(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_loss, wait, best_state = float('inf'), 0, None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=pos_weight).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(X_te.to(device)).cpu().numpy().ravel()
        preds = (logits > 0).astype(int)
        val_bal_acc = balanced_accuracy_score(y_te_np, preds)
        val_loss = -val_bal_acc  # maximize balanced accuracy
        if val_loss < best_loss:
            best_loss, wait = val_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X_te.to(device)).cpu().numpy().ravel()
    preds = (logits > 0).astype(int)
    return {
        'accuracy': float(accuracy_score(y_te_np, preds)),
        'balanced_accuracy': float(balanced_accuracy_score(y_te_np, preds)),
        'f1': float(f1_score(y_te_np, preds, average='binary')),
        'n_train': len(y_train),
        'n_test': len(y_test),
    }


def run_shared_signal_evaluation(reps_train, reps_test, meta_train, meta_test, mode):
    """Evaluate on shared-signal targets: age (Ridge regression) and binarity (LogReg).
    Uses KID column directly if available, otherwise falls back to gaia->KID mapping."""

    if 'KID' in meta_train.columns and meta_train['KID'].notna().any():
        kids_train = meta_train['KID'].fillna(-1).astype(int).values
        kids_test = meta_test['KID'].fillna(-1).astype(int).values
        print(f"  Using KID directly: train={( kids_train > 0).sum()}, "
              f"test={(kids_test > 0).sum()}")
    else:
        # Fallback: gaia_source_id -> KID mapping
        gaia2kid = _build_gaia_to_kid_map()
        kids_train = np.array([gaia2kid.get(g, -1) for g in meta_train['gaia_source_id'].values])
        kids_test = np.array([gaia2kid.get(g, -1) for g in meta_test['gaia_source_id'].values])
        print(f"  gaia->KID mapping: train={(kids_train > 0).sum()}/{len(kids_train)}, "
              f"test={(kids_test > 0).sum()}/{len(kids_test)}")

    age_train = _load_age_labels(kids_train)
    age_test = _load_age_labels(kids_test)
    bin_train = _load_binarity_labels(kids_train)
    bin_test = _load_binarity_labels(kids_test)

    n_age_tr = (~np.isnan(age_train)).sum()
    n_age_te = (~np.isnan(age_test)).sum()
    n_bin_tr = (~np.isnan(bin_train)).sum()
    n_bin_te = (~np.isnan(bin_test)).sum()
    print(f"  Shared-signal labels: age train={n_age_tr} test={n_age_te}, "
          f"binarity train={n_bin_tr} test={n_bin_te}")

    rep_configs = _build_rep_configs(mode)

    all_results = {}
    for rep_name, rep_key in rep_configs.items():
        X_tr, X_te = _get_features(reps_train, reps_test, rep_key)
        if X_tr is None:
            continue

        # Age regression
        mask_tr_age = ~np.isnan(age_train)
        mask_te_age = ~np.isnan(age_test)
        if mask_tr_age.sum() >= 50 and mask_te_age.sum() >= 20:
            scaler_x = StandardScaler().fit(X_tr[mask_tr_age])
            X_tr_s = scaler_x.transform(X_tr[mask_tr_age])
            X_te_s = scaler_x.transform(X_te[mask_te_age])
            model = Ridge(alpha=1.0).fit(X_tr_s, age_train[mask_tr_age])
            y_pred = model.predict(X_te_s)
            all_results[f'{rep_name}/age'] = {
                'r2': float(r2_score(age_test[mask_te_age], y_pred)),
                'rmse': float(np.sqrt(mean_squared_error(age_test[mask_te_age], y_pred))),
                'n_train': int(mask_tr_age.sum()),
                'n_test': int(mask_te_age.sum()),
            }

        # Binarity classification
        mask_tr_bin = ~np.isnan(bin_train)
        mask_te_bin = ~np.isnan(bin_test)
        if mask_tr_bin.sum() >= 50 and mask_te_bin.sum() >= 20:
            res = classification_probe(
                X_tr[mask_tr_bin], bin_train[mask_tr_bin],
                X_te[mask_te_bin], bin_test[mask_te_bin],
            )
            all_results[f'{rep_name}/binarity'] = res

    # Supervised ceiling: MLP trained directly on labels using frozen concat features
    X_tr_cat, X_te_cat = _get_features(reps_train, reps_test, ('z_a', 'z_b'))
    if X_tr_cat is not None:
        mask_tr_age = ~np.isnan(age_train)
        mask_te_age = ~np.isnan(age_test)
        if mask_tr_age.sum() >= 50 and mask_te_age.sum() >= 20:
            print("  Computing supervised ceiling (age)...")
            res = supervised_ceiling_regression(
                X_tr_cat[mask_tr_age], age_train[mask_tr_age],
                X_te_cat[mask_te_age], age_test[mask_te_age])
            res['n_train'] = int(mask_tr_age.sum())
            res['n_test'] = int(mask_te_age.sum())
            all_results['supervised_ceiling/age'] = res

        mask_tr_bin = ~np.isnan(bin_train)
        mask_te_bin = ~np.isnan(bin_test)
        if mask_tr_bin.sum() >= 50 and mask_te_bin.sum() >= 20:
            print("  Computing supervised ceiling (binarity)...")
            res = supervised_ceiling_classification(
                X_tr_cat[mask_tr_bin], bin_train[mask_tr_bin],
                X_te_cat[mask_te_bin], bin_test[mask_te_bin])
            all_results['supervised_ceiling/binarity'] = res

    return all_results


# ============================================================================
# Plotting
# ============================================================================

def plot_training_curves(history, mode, save_path):
    epochs = [h['epoch'] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [h['train']['loss'] for h in history], label='Train', color='#1f77b4')
    ax.plot(epochs, [h['val']['loss'] for h in history], label='Val', color='#ff7f0e')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title(f'Training Curves -- {mode.upper()}')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_probe_comparison(results_by_mode, label_names, save_path):
    fig, axes = plt.subplots(1, len(label_names), figsize=(5 * len(label_names), 5))
    if len(label_names) == 1:
        axes = [axes]
    for i, label in enumerate(label_names):
        ax = axes[i]
        bar_data = {}
        for mode, results in results_by_mode.items():
            for key, val in results.items():
                rep_name, lbl = key.rsplit('/', 1)
                if lbl == label:
                    bar_data[f'{mode}\n{rep_name}'] = val['r2']
        if bar_data:
            names, vals = list(bar_data.keys()), list(bar_data.values())
            colors = ['#4A90C4' if 'ca' in n.lower() else '#E07B54' if 'cp' in n.lower()
                       else '#888888' for n in names]
            ax.barh(names, vals, color=colors, edgecolor='white')
            ax.set_xlabel('R^2'); ax.set_title(label)
            ax.set_xlim(left=min(0, min(vals) - 0.05))
    plt.suptitle('Linear Probe: CA vs CP (LAMOST x TESS)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main training pipeline
# ============================================================================

def run_training(mode, args, *,
                 train_loader=None, val_loader=None, test_loader=None,
                 lamost_enc=None, tess_enc=None,
                 train_data=None, val_data=None, test_data=None,
                 label_names=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}\nTraining mode: {mode.upper()}\nDevice: {device}\n{'='*60}")

    # Detect feature dimensions from first batch
    if train_loader is not None:
        sample_batch = next(iter(train_loader))
        with torch.no_grad():
            z_a, z_b = extract_features_batch(lamost_enc, tess_enc, sample_batch, device)
        dim_a, dim_b = z_a.shape[1], z_b.shape[1]
    else:
        dim_a, dim_b = train_data['z_lamost'].shape[1], train_data['z_tess'].shape[1]
        label_cols = [c for c in label_names if c in train_data['labels'].columns]
        y_train = train_data['labels'][label_cols].values.astype(np.float32)
        y_val = val_data['labels'][label_cols].values.astype(np.float32)
        y_test = test_data['labels'][label_cols].values.astype(np.float32)
        train_loader = DataLoader(
            PairedFeatureDataset(train_data['z_lamost'], train_data['z_tess'], y_train),
            batch_size=args.batch_size, shuffle=True, drop_last=False)
        val_loader = DataLoader(
            PairedFeatureDataset(val_data['z_lamost'], val_data['z_tess'], y_val),
            batch_size=args.batch_size)
        test_loader = DataLoader(
            PairedFeatureDataset(test_data['z_lamost'], test_data['z_tess'], y_test),
            batch_size=args.batch_size)

    print(f"  dim_a (LAMOST): {dim_a}, dim_b (TESS): {dim_b}")

    # Build CA/CP heads
    models = {}
    if mode in ('ca', 'joint'):
        models['proj_a'] = ProjectionHead(dim_a, args.proj_hidden, args.proj_dim,
                                          num_layers=args.proj_layers).to(device)
        models['proj_b'] = ProjectionHead(dim_b, args.proj_hidden, args.proj_dim,
                                          num_layers=args.proj_layers).to(device)
    if mode in ('cp', 'joint'):
        models['predictor_a2b'] = CrossPredictor(dim_a, dim_b, args.pred_hidden,
                                                  num_layers=args.pred_layers,
                                                  dropout=args.dropout).to(device)
    if mode == 'cp_reverse':
        models['predictor_b2a'] = CrossPredictor(dim_b, dim_a, args.pred_hidden,
                                                  num_layers=args.pred_layers,
                                                  dropout=args.dropout).to(device)

    n_params = sum(p.numel() for m in models.values() for p in m.parameters())
    print(f"  Trainable parameters: {n_params:,}")

    all_params = [p for m in models.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_m = train_epoch(mode, train_loader, models, optimizer, device,
                              args.ca_loss, args.joint_alpha, lamost_enc, tess_enc)
        val_m = validate(mode, val_loader, models, device,
                         args.ca_loss, args.joint_alpha, lamost_enc, tess_enc)
        scheduler.step()
        history.append({'epoch': epoch, 'train': train_m, 'val': val_m})

        if epoch % args.print_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_m['loss']:.4f} | val={val_m['loss']:.4f}")

        if val_m['loss'] < best_val_loss:
            best_val_loss = val_m['loss']
            patience_counter = 0
            best_state = {k: m.state_dict() for k, m in models.items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state:
        for k, m in models.items():
            m.load_state_dict(best_state[k])

    # Extract reps & evaluate -- use non-shuffled loader for train
    if train_data is not None:
        train_loader_eval = DataLoader(
            PairedFeatureDataset(train_data['z_lamost'], train_data['z_tess'], y_train),
            batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        train_loader_eval = train_loader
    reps_train = extract_representations(mode, train_loader_eval, models, device,
                                         lamost_enc, tess_enc)
    reps_test = extract_representations(mode, test_loader, models, device,
                                        lamost_enc, tess_enc)
    if label_names is None:
        label_names = EVAL_LABELS
    eval_results = run_evaluation(reps_train, reps_test, label_names, mode)

    print(f"\n  Linear Probe Results ({mode.upper()}):")
    for key, val in sorted(eval_results.items()):
        print(f"    {key}: R2={val['r2']:.4f}, RMSE={val['rmse']:.4f} (n={val['n_test']})")

    # Shared-signal evaluation (age + binarity)
    shared_results = {}
    if train_data is not None and 'metadata' in train_data:
        print(f"\n  Shared-Signal Evaluation ({mode.upper()}):")
        shared_results = run_shared_signal_evaluation(
            reps_train, reps_test,
            train_data['metadata'], test_data['metadata'], mode)
        for key, val in sorted(shared_results.items()):
            if 'r2' in val:
                print(f"    {key}: R2={val['r2']:.4f}, RMSE={val['rmse']:.4f} (n={val['n_test']})")
            else:
                print(f"    {key}: acc={val['accuracy']:.4f}, bal_acc={val['balanced_accuracy']:.4f}, "
                      f"f1={val['f1']:.4f} (n={val['n_test']})")
        eval_results.update(shared_results)

    return {
        'mode': mode, 'history': history, 'eval_results': eval_results,
        'best_val_loss': best_val_loss, 'n_params': n_params,
        'models_state': best_state,
    }


# ============================================================================
# Data preparation helpers
# ============================================================================

def build_transforms():
    """Build LAMOST + TESS transform pipelines using MultiDESA transforms."""
    sys.path.insert(0, str(MULTIDESA_ROOT))
    from transforms.transforms import (Compose, GeneralSpectrumPreprocessor,
                                        Normalize, ACF, FFT, LombScargle,
                                        SigmaClip)

    lamost_transforms = Compose([
        GeneralSpectrumPreprocessor(rv_norm=True, continuum_norm=True),
        Normalize(scheme=['std'], axis=0),
    ])

    tess_transforms = Compose([
        SigmaClip(sigma_upper=3.0, sigma_lower=10.0, max_iters=3, replace_with='median'),
        Normalize(scheme=['std'], axis=0),
        ACF(normalize=True),
        FFT(normalize=True),
        LombScargle(max_freq=2.0, normalize=True),
    ])
    return lamost_transforms, tess_transforms


def _filter_and_split(both, seed=42, target_total=40000):
    """Build ~40k sample dataset with strategic splitting:
    - Binarity stars -> train/val/test (70/15/15) so probe can train
    - Prot-only stars (no binarity) -> train/val/test (70/15/15)
    - Random padding stars (no prot, no binarity) -> train/val only
    - All stars must have valid logg
    """
    both = both[both['logg'].notna()].copy()

    prot_col = 'prot' if 'prot' in both.columns else ('Prot' if 'Prot' in both.columns else None)
    prot_mask = both[prot_col].notna() if prot_col else pd.Series(False, index=both.index)

    nss_df = pd.read_csv(NSS_CATALOG, low_memory=False)
    nss_df = nss_df[nss_df['binarity_class'] != 3]
    nss_kids = set(nss_df['KID'].values)
    bin_mask = both['KID'].apply(lambda k: int(k) in nss_kids if pd.notna(k) else False)

    df_bin = both[bin_mask]
    df_prot_only = both[prot_mask & ~bin_mask]  # prot without binarity -> split
    df_neither = both[~prot_mask & ~bin_mask]   # padding pool

    print(f"  Binarity (-> train/val/test): {len(df_bin)}")
    print(f"  Prot-only (-> train/val/test): {len(df_prot_only)}")
    print(f"  Padding pool: {len(df_neither)}")

    rng = np.random.RandomState(seed)

    # Split binarity into train/val/test (70/15/15)
    idx_b = rng.permutation(len(df_bin))
    n_train_b = int(len(df_bin) * 0.7)
    n_val_b = int(len(df_bin) * 0.15)
    bin_train = df_bin.iloc[idx_b[:n_train_b]]
    bin_val = df_bin.iloc[idx_b[n_train_b:n_train_b + n_val_b]]
    bin_test = df_bin.iloc[idx_b[n_train_b + n_val_b:]]

    # Split prot-only into train/val/test (70/15/15)
    idx = rng.permutation(len(df_prot_only))
    n_train_p = int(len(df_prot_only) * 0.7)
    n_val_p = int(len(df_prot_only) * 0.15)

    prot_train = df_prot_only.iloc[idx[:n_train_p]]
    prot_val = df_prot_only.iloc[idx[n_train_p:n_train_p + n_val_p]]
    prot_test = df_prot_only.iloc[idx[n_train_p + n_val_p:]]

    # Random padding to reach target_total
    n_random = target_total - len(df_prot_only) - len(df_bin)
    n_random = max(0, min(n_random, len(df_neither)))
    df_random = df_neither.sample(n=n_random, random_state=seed)

    # Split random into train/val only (proportional to train:val ratio = 70:15)
    idx_r = rng.permutation(len(df_random))
    n_rand_train = int(len(df_random) * (0.7 / 0.85))
    rand_train = df_random.iloc[idx_r[:n_rand_train]]
    rand_val = df_random.iloc[idx_r[n_rand_train:]]

    df_train = pd.concat([prot_train, bin_train, rand_train])
    df_val = pd.concat([prot_val, bin_val, rand_val])
    df_test = pd.concat([prot_test, bin_test])

    print(f"  train: {len(df_train)} ({len(prot_train)} prot + {len(bin_train)} bin + {len(rand_train)} random)")
    print(f"  val:   {len(df_val)} ({len(prot_val)} prot + {len(bin_val)} bin + {len(rand_val)} random)")
    print(f"  test:  {len(df_test)} ({len(prot_test)} prot + {len(bin_test)} bin)")
    print(f"  total: {len(df_train) + len(df_val) + len(df_test)}")
    return df_train, df_val, df_test


def build_on_the_fly_loaders(config, args):
    """Build train/val/test DataLoaders from raw data."""
    print("Building on-the-fly data loaders from raw LAMOST HDF5 + TESS HDF5 ...")
    df = pd.read_parquet(config['crossmatch_parquet'])
    both = df[df['has_lamost'] & df['has_tess']].copy()

    # Deduplicate by star (keep highest LAMOST SNR)
    both = both.sort_values('lamost_snr', ascending=False).drop_duplicates('gaia_source_id')
    print(f"  {len(both)} unique paired stars")

    # Filter and split (binarity-only -> test)
    df_train, df_val, df_test = _filter_and_split(both, seed=args.seed)

    lamost_transforms, tess_transforms = build_transforms()

    def _make_loader(df_split, shuffle):
        ds = CrossModalPairDataset(
            df_split,
            lamost_hdf5_dir=config['lamost_hdf5_dir'],
            lamost_transforms=lamost_transforms,
            tess_transforms=tess_transforms,
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, drop_last=shuffle,
                          pin_memory=True, persistent_workers=args.num_workers > 0)

    return _make_loader(df_train, True), _make_loader(df_val, False), _make_loader(df_test, False)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Cross-Modal CA vs CP: LAMOST x TESS')
    parser.add_argument('--mode', type=str, default='all',
                        choices=['ca', 'cp', 'cp_reverse', 'joint', 'all'])
    parser.add_argument('--use_cached_features', action='store_true',
                        help='Use pre-extracted features instead of on-the-fly encoding')

    # Architecture
    parser.add_argument('--proj_dim', type=int, default=256)
    parser.add_argument('--proj_hidden', type=int, default=512)
    parser.add_argument('--proj_layers', type=int, default=2)
    parser.add_argument('--pred_hidden', type=int, default=512)
    parser.add_argument('--pred_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--ca_loss', type=str, default='vicreg',
                        choices=['vicreg', 'infonce'])
    parser.add_argument('--joint_alpha', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=4)

    # Data
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Subsample the paired catalog (useful for on-the-fly mode)')
    parser.add_argument('--eval_labels', nargs='+', default=EVAL_LABELS)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--print_every', type=int, default=10)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if args.output_dir is None:
        args.output_dir = os.path.join(DEFAULT_CONFIG['output_dir'], timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump({**vars(args), 'phase_params': PHASE_PARAMS}, f, indent=2, default=str)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Data setup --
    lamost_enc = tess_enc = None
    train_loader = val_loader = test_loader = None
    train_data = val_data = test_data = None

    if args.use_cached_features:
        cache_dir = os.path.join(os.path.dirname(DEFAULT_CONFIG['output_dir']), 'cached_features_tess')
        if os.path.isdir(cache_dir) and os.path.exists(os.path.join(cache_dir, 'z_lamost_train.npy')):
            print(f"\nLoading pre-extracted features from {cache_dir}")
            train_data, val_data, test_data = load_cached_splits(cache_dir)
        else:
            raise FileNotFoundError(
                f"Cached features not found at {cache_dir}. "
                f"Run extract_cross_modal_features_tess.py first.")
    else:
        train_loader, val_loader, test_loader = build_on_the_fly_loaders(
            DEFAULT_CONFIG, args)
        print("\nLoading frozen pretrained encoders...")
        lamost_enc = build_lamost_encoder(DEFAULT_CONFIG, device)
        tess_enc = build_tess_encoder(DEFAULT_CONFIG, device)

    # -- Training --
    modes = ['ca', 'cp', 'cp_reverse'] if args.mode == 'all' else [args.mode]
    all_results = {}

    for mode in modes:
        if args.use_cached_features:
            result = run_training(
                mode, args,
                train_data=train_data, val_data=val_data, test_data=test_data,
                label_names=args.eval_labels)
        else:
            result = run_training(
                mode, args,
                train_loader=train_loader, val_loader=val_loader,
                test_loader=test_loader,
                lamost_enc=lamost_enc, tess_enc=tess_enc,
                label_names=args.eval_labels)

        all_results[mode] = result
        mode_dir = os.path.join(args.output_dir, mode)
        os.makedirs(mode_dir, exist_ok=True)
        plot_training_curves(result['history'], mode,
                             os.path.join(mode_dir, 'training_curves.png'))
        torch.save(result['models_state'], os.path.join(mode_dir, 'best_models.pth'))
        saveable = {k: v for k, v in result.items() if k != 'models_state'}
        with open(os.path.join(mode_dir, 'results.json'), 'w') as f:
            json.dump(saveable, f, indent=2, default=str)

    # Combined plot
    if len(all_results) > 1:
        eval_by_mode = {m: r['eval_results'] for m, r in all_results.items()}
        avail = [l for l in args.eval_labels
                 if any(l in k for res in eval_by_mode.values() for k in res)]
        if avail:
            plot_probe_comparison(eval_by_mode, avail,
                                  os.path.join(args.output_dir, 'probe_comparison.png'))

    # Summary
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"Phase diagram prediction: {PHASE_PARAMS['predicted_regime']}")
    print(f"  Delta_CA = {PHASE_PARAMS['Delta_CA']:.3f} (<1 -> CA fails)")
    print(f"  Delta_CP = {PHASE_PARAMS['Delta_CP']:.3f} (<1 -> CP fails too)\n")
    for mode, result in all_results.items():
        print(f"{mode.upper()}: best_val_loss={result['best_val_loss']:.4f}")
        for key, val in sorted(result['eval_results'].items()):
            if 'r2' in val:
                print(f"  {key}: R2={val['r2']:.4f}, RMSE={val.get('rmse', 0):.4f}")
            elif 'accuracy' in val:
                print(f"  {key}: acc={val['accuracy']:.4f}, bal_acc={val['balanced_accuracy']:.4f}")
    print()

    summary = {
        'phase_prediction': PHASE_PARAMS,
        'data_mode': 'cached' if args.use_cached_features else 'on_the_fly',
        'results': {m: {k: v for k, v in r.items() if k != 'models_state'}
                    for m, r in all_results.items()},
    }
    with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"All results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
