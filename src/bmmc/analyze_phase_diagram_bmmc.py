"""
Phase diagram analysis for BMMC CITE-seq (RNA × ADT).

Features:
  Z1 = scGPT RNA embeddings  (90243 × 512, cached .npy)
  Z2 = raw CLR-normalized ADT (90243 × 134, from h5ad)
       OR pretrained ADT encoder embeddings (90243 × 512) via --adt_pretrained

Labels:
  signal = cell_type  (44 classes)

Subsamples to --max_cells (default 20000) for speed.
"""
from __future__ import annotations

import sys
import json
import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
MM_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MM_ROOT / 'src'))
sys.path.insert(0, str(MM_ROOT / 'src/bmmc'))

RNA_EMB  = Path('/home/ilay.kamai/work/citeseq/bmmc/rna_embeddings_scgpt.npy')
H5AD     = Path('/home/ilay.kamai/work/citeseq/bmmc/neurips2021_cite_BMMC.h5ad')
OUT_DIR  = MM_ROOT / 'src/logs/bmmc/phase_diagram'

# ── import analysis functions ───────────────────────────────────────────────
from analyze_phase_diagram import (
    estimate_parameters,
    plot_diagnostic,
)
from sklearn.decomposition import PCA


def _prereduce_and_center(Z1, Z2, pca_prereduce):
    k1 = min(pca_prereduce, Z1.shape[1], Z1.shape[0] - 1)
    k2 = min(pca_prereduce, Z2.shape[1], Z2.shape[0] - 1)
    Z1c = PCA(n_components=k1, random_state=0).fit_transform(Z1)
    Z2c = PCA(n_components=k2, random_state=0).fit_transform(Z2)
    Z1c -= Z1c.mean(axis=0)
    Z2c -= Z2c.mean(axis=0)
    return Z1c, Z2c


def one_hot(values, classes):
    c2i = {c: i for i, c in enumerate(classes)}
    arr = np.zeros((len(values), len(classes)), dtype=np.float32)
    for row, v in enumerate(values):
        arr[row, c2i[v]] = 1.0
    return arr


def load_data(adt_pretrained=None):
    """Load and align Z1 (RNA), Z2 (ADT), labels. Returns (Z1, Z2, labels, label_names, z2_label).

    If adt_pretrained is a path, Z2 is the pretrained ADTEncoder's embeddings (512-dim)
    instead of raw CLR-normalized ADT (134-dim).
    """
    import anndata
    print("Loading BMMC h5ad...")
    adata = anndata.read_h5ad(str(H5AD))
    print(f"  Full: {adata.shape}")

    print("Loading cached scGPT RNA embeddings...")
    Z1_full = np.load(str(RNA_EMB)).astype(np.float32)
    print(f"  Z1 (RNA): {Z1_full.shape}")

    print("Loading ADT via load_bmmc()...")
    from bmmc_dataset import load_bmmc
    rna_adata, adt_adata, obs_df = load_bmmc(str(H5AD))
    X_adt = adt_adata.X
    if sp.issparse(X_adt):
        X_adt = X_adt.toarray()
    X_adt = X_adt.astype(np.float32)
    print(f"  ADT raw: {X_adt.shape}")

    n_full_adata = adata.shape[0]
    n_filtered   = len(obs_df)

    if Z1_full.shape[0] == n_full_adata:
        filtered_idx = obs_df.index
        orig_idx     = pd.Index(adata.obs_names)
        int_idx      = orig_idx.get_indexer(filtered_idx)
        assert (int_idx >= 0).all()
        Z1_full = Z1_full[int_idx]
        print(f"  Z1 after rare-type filter: {Z1_full.shape}")
    elif Z1_full.shape[0] == n_filtered:
        print(f"  Z1 already filtered: {Z1_full.shape}")
    else:
        raise ValueError(f"Z1 shape mismatch: {Z1_full.shape[0]} vs full={n_full_adata} filtered={n_filtered}")

    assert Z1_full.shape[0] == X_adt.shape[0] == n_filtered

    if adt_pretrained:
        import torch
        from cross_modal_bmmc import ADTEncoder
        print(f"  Loading pretrained ADT encoder from {adt_pretrained}...")
        enc = ADTEncoder(input_dim=X_adt.shape[1], hidden_dims=[256, 512], embed_dim=512)
        enc.load_state_dict(torch.load(adt_pretrained, map_location='cpu'))
        enc.eval()
        with torch.no_grad():
            Z2_full = enc(torch.from_numpy(X_adt)).numpy()
        print(f"  Z2 (ADT pretrained): {Z2_full.shape}")
        z2_label = 'ADT (pretrained encoder)'
    else:
        Z2_full  = X_adt
        z2_label = 'ADT (CLR)'
        print(f"  Z2 (ADT raw): {Z2_full.shape}")

    cell_types = obs_df['cell_type'].values
    ct_classes = sorted(set(cell_types))
    print(f"  Cell types: {len(ct_classes)}")
    labels      = one_hot(cell_types, ct_classes)
    label_names = [f'ct:{c}' for c in ct_classes]

    return Z1_full, Z2_full, labels, label_names, z2_label


def run(Z1_full, Z2_full, labels, label_names,
        max_cells=1000, pca_components=100, n_components=100,
        n_permutations=30, seed=42, out_dir=None, z2_label='ADT (CLR)'):
    """Run phase estimation for one (max_cells, seed) configuration."""
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # subsample
    N = Z1_full.shape[0]
    if max_cells > 0 and N > max_cells:
        idx = rng.choice(N, max_cells, replace=False)
        idx.sort()
        Z1 = Z1_full[idx]
        Z2 = Z2_full[idx]
        lbl = labels[idx]
        print(f"[n={max_cells} s={seed}] Subsampled {max_cells}/{N}")
    else:
        Z1, Z2, lbl = Z1_full, Z2_full, labels

    # PCA + center
    Z1c, Z2c = _prereduce_and_center(Z1, Z2, pca_components)

    n_comp = min(n_components, Z1c.shape[1], Z2c.shape[1])
    print(f"[n={max_cells} s={seed}] Estimating (n_comp={n_comp})...")
    result = estimate_parameters(
        Z1c, Z2c,
        labels=lbl,
        n_components=n_comp,
        cv=5,
        n_permutations=n_permutations,
        classification_method='elbow',
    )

    dca = result['delta_ca_direct']
    dcp = result['delta_cp_direct']
    reg = result['predicted_regime']
    print(f"[n={max_cells} s={seed}] Δ_CA={dca:.3f}  Δ_CP={dcp:.3f}  => {reg}")

    # save NPZ
    np.savez(
        str(out_dir / 'bmmc_classification_data.npz'),
        cca_sv              = result['cca']['canonical_correlations'],
        cca_r2              = result['cca']['r2_matrix'],
        cca_is_signal       = result['cca']['is_signal'],
        cca_n_signal        = result['cca']['n_signal'],
        cp_direct_sv        = result['cp_direct']['singular_values'],
        cp_direct_sum_r2    = result['cp_direct']['sum_r2'],
        cp_direct_is_signal = result['cp_direct']['is_signal'],
        cp_direct_n_signal  = result['cp_direct']['n_signal'],
    )

    # save JSON
    serializable = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.items() if not isinstance(v, dict)}
    with open(out_dir / 'bmmc_result.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    # diagnostic plot
    plot_diagnostic(result, 'RNA (scGPT)', z2_label,
                    label_names=label_names,
                    out_path=str(out_dir / 'bmmc_diagnostic.png'))

    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--max_cells', type=int, default=20000)
    p.add_argument('--pca_components', type=int, default=100)
    p.add_argument('--n_components', type=int, default=100)
    p.add_argument('--n_permutations', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', type=str, default=None)
    p.add_argument('--adt_pretrained', type=str, default=None,
                   help='Path to pretrained ADTEncoder state dict (.pt). '
                        'If set, Z2 = encoder embeddings (512-dim) instead of raw CLR ADT.')
    return p.parse_args()


def main():
    args = parse_args()
    Z1, Z2, labels, label_names, z2_label = load_data(adt_pretrained=args.adt_pretrained)
    default_out = (MM_ROOT / 'src/logs/bmmc_pretrained/phase_diagram'
                   if args.adt_pretrained else OUT_DIR)
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    run(Z1, Z2, labels, label_names,
        max_cells=args.max_cells,
        pca_components=args.pca_components,
        n_components=args.n_components,
        n_permutations=args.n_permutations,
        seed=args.seed,
        out_dir=out_dir,
        z2_label=z2_label)


if __name__ == '__main__':
    main()
