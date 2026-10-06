"""
Floor attribution for the BMMC CITE-seq recovery-regime estimate.

The direct-spectral estimator (src/analyze_phase_diagram.py) reports for BMMC
(RNA/scGPT x ADT/CLR, 44-class cell-type label, n=5000):

    Delta_CA = min(phi_CCA[signal]) / max(phi_CCA[nuisance]) ~ 0.64
    Delta_CP = min(sigma_A[signal]) / max(sigma_A[nuisance]) ~ 0.52   => "Neither"

Delta is a ratio of extreme order statistics; the DENOMINATOR (nuisance floor)
is a max set by a SINGLE component.  This script identifies the floor-setting
component(s) in each decomposition and characterizes them by regressing their
projected score vectors against every available cell-level covariate OTHER than
cell type, using the SAME 5-fold Ridge CV R^2 protocol as the classifier.

It does NOT modify the estimator, the elbow classifier, or any thresholds.
It imports estimate_parameters / plot_diagnostic / the R^2 helpers read-only and
reproduces the CCA and A-SVD decompositions to recover the per-component scores
(which the estimator does not return).

Stages:
  exp1  : reproduce estimate, identify floor + reference components, attribute.
  exp2  : (conditional) re-estimate after (a) covariate regress-out and
          (b) within-donor centering; emit before/after Delta table + panels.

Usage:
  python -m src.bmmc.floor_attribution --stage exp1
  python -m src.bmmc.floor_attribution --stage exp2
"""
from __future__ import annotations

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
from sklearn.decomposition import PCA

MM_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MM_ROOT / 'src'))

from analyze_phase_diagram import (           # read-only imports
    estimate_parameters,
    plot_diagnostic,
    _make_cv_folds,
    _r2_ridge_2d,
)

# ── paths (mirror analyze_phase_diagram_bmmc.py) ─────────────────────────────
H5AD    = Path('/home/ilay.kamai/work/citeseq/bmmc/neurips2021_cite_BMMC.h5ad')
RNA_EMB = Path('/home/ilay.kamai/work/citeseq/bmmc/rna_embeddings_scgpt.npy')
OUT_DIR = MM_ROOT / 'src/logs/bmmc/floor_attribution'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── estimation config (exactly bmmc_phase.sbatch defaults) ───────────────────
MAX_CELLS      = 5000
PCA_COMPONENTS = 100
N_COMPONENTS   = 100
N_PERMUTATIONS = 30
SEED           = 42
MIN_CELLS_PER_TYPE = 20
CCA_REG = 1e-6


# ═════════════════════════════════════════════════════════════════════════
# Data loading (h5py only — no anndata/scanpy needed; embeddings are cached)
# ═════════════════════════════════════════════════════════════════════════

def _decode(arr):
    return np.array([x.decode() if isinstance(x, (bytes, bytearray)) else x for x in arr])


def _cat(f, col):
    """Decode an obs categorical column to string labels (full 90261 order)."""
    codes = f[f'obs/{col}'][:]
    cats  = _decode(f[f'obs/__categories/{col}'][:])
    return cats[codes], cats


def _extract_adt_rows(f, adt_cols, orig_rows):
    """Extract dense ADT (len(orig_rows) x n_adt) from CSR X for given rows."""
    Xg = f['X']
    data, indices, indptr = Xg['data'], Xg['indices'], Xg['indptr']
    col_pos = {c: j for j, c in enumerate(adt_cols)}
    out = np.zeros((len(orig_rows), len(adt_cols)), dtype=np.float32)
    for i, r in enumerate(orig_rows):
        s, e = int(indptr[r]), int(indptr[r + 1])
        cols = indices[s:e]
        vals = data[s:e]
        for c, v in zip(cols, vals):
            j = col_pos.get(int(c))
            if j is not None:
                out[i, j] = v
    return out


def load_bmmc_subsample():
    """
    Reproduce exactly the (Z1, Z2, labels, covariates) fed to estimate_parameters
    by analyze_phase_diagram_bmmc.py at max_cells=5000, seed=42.

    Returns dict with raw (pre-PCA) Z1/Z2 (n x 512 / n x 134), one-hot cell-type
    labels, class names, and a covariate DataFrame aligned to the subsample.
    """
    print("Loading BMMC (h5py, direct)...")
    f = h5py.File(str(H5AD), 'r')

    # feature_types → ADT column indices (var order preserved, as in load_bmmc)
    ft, _ = None, None
    ft_codes = f['var/feature_types'][:]
    ft_cats  = _decode(f['var/__categories/feature_types'][:])
    ft_names = ft_cats[ft_codes]
    adt_cols = np.where(ft_names == 'ADT')[0]
    assert len(adt_cols) == 134, f"expected 134 ADT, got {len(adt_cols)}"

    # cell_type (full) + rare-type filter (>=20 cells), order-preserving
    cell_type_full, ct_cats = _cat(f, 'cell_type')
    vc = pd.Series(cell_type_full).value_counts()
    valid_types = set(vc[vc >= MIN_CELLS_PER_TYPE].index)
    mask = np.array([ct in valid_types for ct in cell_type_full])
    orig_idx_filtered = np.where(mask)[0]          # original rows of filtered set
    n_filtered = len(orig_idx_filtered)
    print(f"  filtered {mask.sum()}/{len(mask)} cells; "
          f"{len(valid_types)}/{len(ct_cats)} cell types kept")

    # cached scGPT embeddings — already in filtered order (90243 x 512)
    Z1_all = np.load(str(RNA_EMB)).astype(np.float32)
    assert Z1_all.shape[0] == n_filtered, \
        f"Z1 {Z1_all.shape[0]} != filtered {n_filtered}"

    # subsample: seed 42, 5000, sorted (exactly the bmmc script)
    rng = np.random.default_rng(SEED)
    if n_filtered > MAX_CELLS:
        sub = rng.choice(n_filtered, MAX_CELLS, replace=False)
        sub.sort()
    else:
        sub = np.arange(n_filtered)
    orig_sel = orig_idx_filtered[sub]              # original rows of subsample
    print(f"  subsample: {len(sub)} cells")

    Z1 = Z1_all[sub]
    Z2 = _extract_adt_rows(f, adt_cols, orig_sel)   # (n x 134) CLR ADT
    cell_type_sel = cell_type_full[orig_sel]

    # one-hot cell-type labels over the 44 kept classes (sorted), as bmmc does
    ct_classes = sorted(valid_types)
    c2i = {c: i for i, c in enumerate(ct_classes)}
    labels = np.zeros((len(sub), len(ct_classes)), dtype=np.float32)
    for r, ct in enumerate(cell_type_sel):
        labels[r, c2i[ct]] = 1.0

    # ── covariates (aligned to subsample) ────────────────────────────────
    cov = pd.DataFrame(index=np.arange(len(sub)))
    cov['cell_type'] = cell_type_sel
    for col in ['DonorNumber', 'batch', 'Site', 'GEX_phase', 'Samplename']:
        cov[col] = _cat(f, col)[0][orig_sel]
    for col in ['GEX_size_factors', 'GEX_n_genes_by_counts', 'GEX_pct_counts_mt',
                'ADT_total_counts', 'ADT_n_antibodies_by_counts', 'ADT_iso_count']:
        cov[col] = np.asarray(f[f'obs/{col}'][:])[orig_sel]
    f.close()

    return {
        'Z1': Z1, 'Z2': Z2, 'labels': labels,
        'ct_classes': ct_classes, 'cov': cov,
        'orig_sel': orig_sel,
    }


def prereduce_and_center(Z1, Z2, k=PCA_COMPONENTS):
    """Exactly analyze_phase_diagram_bmmc._prereduce_and_center."""
    k1 = min(k, Z1.shape[1], Z1.shape[0] - 1)
    k2 = min(k, Z2.shape[1], Z2.shape[0] - 1)
    Z1c = PCA(n_components=k1, random_state=0).fit_transform(Z1)
    Z2c = PCA(n_components=k2, random_state=0).fit_transform(Z2)
    Z1c -= Z1c.mean(axis=0)
    Z2c -= Z2c.mean(axis=0)
    return Z1c, Z2c


# ═════════════════════════════════════════════════════════════════════════
# Reproduce CCA and A-SVD decompositions to recover per-component scores
# (mirrors _cca_analysis / _cp_direct_analysis exactly; read-only re-derivation)
# ═════════════════════════════════════════════════════════════════════════

def _eigh_inv_sqrt(M):
    ev, V = np.linalg.eigh(M)
    ev = np.maximum(ev, 1e-12)
    return V @ np.diag(1.0 / np.sqrt(ev)) @ V.T


def cca_scores(Z1, Z2, reg=CCA_REG, nc=N_COMPONENTS):
    """Return phi, RNA-side scores (n x nc), ADT-side scores (n x nc)."""
    n = Z1.shape[0]
    Z1_c, Z2_c = Z1 - Z1.mean(0), Z2 - Z2.mean(0)
    Sxx = Z1_c.T @ Z1_c / n + reg * np.eye(Z1.shape[1])
    Syy = Z2_c.T @ Z2_c / n + reg * np.eye(Z2.shape[1])
    Sxy = Z1_c.T @ Z2_c / n
    Sx_i, Sy_i = _eigh_inv_sqrt(Sxx), _eigh_inv_sqrt(Syy)
    M = Sx_i @ Sxy @ Sy_i
    P, phi, Qt = np.linalg.svd(M, full_matrices=False)
    nc = min(nc, len(phi))
    phi = np.clip(phi[:nc], 0, 1)
    W_x = Sx_i @ P[:, :nc]
    W_y = Sy_i @ Qt.T[:, :nc]
    return phi, Z1_c @ W_x, Z2_c @ W_y


def asvd_scores(Z1, Z2, reg=CCA_REG, nc=N_COMPONENTS):
    """A = Sigma_yx Sigma_xx^{-1/2}. Return sigma_A, source-side scores (n x nc)."""
    n = Z1.shape[0]
    Z1_c, Z2_c = Z1 - Z1.mean(0), Z2 - Z2.mean(0)
    Sxx = Z1_c.T @ Z1_c / n + reg * np.eye(Z1.shape[1])
    Syx = Z2_c.T @ Z1_c / n
    Sx_i = _eigh_inv_sqrt(Sxx)
    A = Syx @ Sx_i
    U, sigma, Vt = np.linalg.svd(A, full_matrices=False)
    nc = min(nc, len(sigma))
    sigma = sigma[:nc]
    W_src = Sx_i @ Vt.T[:, :nc]
    return sigma, Z1_c @ W_src


# ═════════════════════════════════════════════════════════════════════════
# Covariate design matrices + R^2 attribution (same protocol as classifier)
# ═════════════════════════════════════════════════════════════════════════

LINEAGE_RULES = [   # (substring, coarse lineage) — first match wins
    ('Plasma', 'B'), ('Plasmablast', 'B'), ('Transitional B', 'B'), ('B1 B', 'B'),
    (' B ', 'B'),
    ('CD4+ T', 'CD4_T'), ('Treg', 'CD4_T'), ('T reg', 'CD4_T'),
    ('CD8+ T', 'CD8_T'),
    ('MAIT', 'other_T'), ('gdT', 'other_T'), ('gd T', 'other_T'), ('dnT', 'other_T'),
    ('ILC', 'ILC'),
    ('NK', 'NK'),
    ('Naive CD20+ B', 'B'),
    ('Mono', 'Myeloid'), ('CD14', 'Myeloid'), ('CD16', 'Myeloid'), ('Macrophage', 'Myeloid'),
    ('cDC', 'DC'), ('pDC', 'DC'), ('DC', 'DC'),
    ('Erythro', 'Erythroid'), ('Normoblast', 'Erythroid'), ('Reticulocyte', 'Erythroid'),
    ('Proerythroblast', 'Erythroid'),
    ('HSC', 'Progenitor'), ('Prog', 'Progenitor'), ('MK/E', 'Progenitor'),
    ('GMP', 'Progenitor'), ('LMPP', 'Progenitor'), ('MPP', 'Progenitor'),
    ('CLP', 'Progenitor'), ('G/M prog', 'Progenitor'), ('Mk/E prog', 'Progenitor'),
]


def coarse_lineage(cell_types):
    out = []
    for ct in cell_types:
        lab = 'other'
        for sub, lin in LINEAGE_RULES:
            if sub in ct:
                lab = lin
                break
        out.append(lab)
    return np.array(out)


def _onehot(series):
    cats = sorted(pd.unique(series))
    c2i = {c: i for i, c in enumerate(cats)}
    X = np.zeros((len(series), len(cats)), dtype=np.float64)
    for r, v in enumerate(series):
        X[r, c2i[v]] = 1.0
    return X


def _zscore(x):
    x = np.asarray(x, dtype=np.float64)
    s = x.std()
    return ((x - x.mean()) / s if s > 1e-12 else x - x.mean()).reshape(-1, 1)


def build_covariates(cov: pd.DataFrame):
    """
    Return ordered list of (name, class, design_matrix), where class is one of
    'technical' (donor/batch/depth/ambient), 'biological' (non-task), 'task'.
    Continuous covariates are z-scored; categoricals are one-hot.
    """
    lin = coarse_lineage(cov['cell_type'].values)
    log = lambda c: np.log(np.asarray(cov[c], float) + 1e-8)
    log1p = lambda c: np.log1p(np.asarray(cov[c], float))
    C = []
    # technical / donor / depth / ambient
    C.append(('donor',            'technical', _onehot(cov['DonorNumber'])))
    C.append(('batch',            'technical', _onehot(cov['batch'])))
    C.append(('site',             'technical', _onehot(cov['Site'])))
    C.append(('rna_log_depth',    'technical', _zscore(log('GEX_size_factors'))))
    C.append(('rna_n_genes',      'technical', _zscore(log1p('GEX_n_genes_by_counts'))))
    C.append(('pct_mito',         'technical', _zscore(cov['GEX_pct_counts_mt'])))
    C.append(('adt_log_total',    'technical', _zscore(log1p('ADT_total_counts'))))
    C.append(('adt_n_antibodies', 'technical', _zscore(cov['ADT_n_antibodies_by_counts'])))
    C.append(('adt_iso_count',    'technical', _zscore(log1p('ADT_iso_count'))))
    # biological non-task
    C.append(('cell_cycle',       'biological', _onehot(cov['GEX_phase'])))
    C.append(('lineage_coarse',   'biological', _onehot(pd.Series(lin))))
    # task reference
    C.append(('cell_type',        'task',       _onehot(cov['cell_type'])))
    return C


def attribute(score, covariates, folds):
    """R^2 of predicting the 1-D score from each covariate (5-fold Ridge CV)."""
    y = np.asarray(score, float).reshape(-1, 1)
    out = {}
    for name, klass, X in covariates:
        r2 = _r2_ridge_2d(X, y, folds)   # shape (1,)
        out[name] = float(r2[0])
    return out


# ═════════════════════════════════════════════════════════════════════════
# Component selection
# ═════════════════════════════════════════════════════════════════════════

BAND_K = 22   # characterize the top-K components by singular value per decomposition


def role_map(spectrum, is_signal):
    """
    Map component index -> special role (floor sets the Delta denominator, etc.).
      floor1/2/3 : top-3 nuisance components by singular value (floor1 = the max)
      numerator  : weakest signal component (min sing value among signal) — sets min_i rho
      top_signal : strongest signal component (max sing value among signal)
    Components not in these roles get role ''.
    """
    sig = np.where(is_signal)[0]
    nui = np.where(~is_signal)[0]
    roles = {}
    nui_sorted = nui[np.argsort(spectrum[nui])[::-1]]
    for rank, j in enumerate(nui_sorted[:3], 1):
        roles[int(j)] = f'floor{rank}_nuisance'
    if len(sig):
        roles.setdefault(int(sig[np.argmin(spectrum[sig])]), 'numerator_weakest_signal')
        roles.setdefault(int(sig[np.argmax(spectrum[sig])]), 'top_signal')
    return roles


# ═════════════════════════════════════════════════════════════════════════
# Experiment 1
# ═════════════════════════════════════════════════════════════════════════

def run_exp1():
    data = load_bmmc_subsample()
    Z1, Z2, labels = data['Z1'], data['Z2'], data['labels']
    Z1c, Z2c = prereduce_and_center(Z1, Z2)
    print(f"  Z1c {Z1c.shape}, Z2c {Z2c.shape}")

    nc = min(N_COMPONENTS, Z1c.shape[1], Z2c.shape[1])
    print("\nReproducing estimate_parameters ...")
    result = estimate_parameters(Z1c, Z2c, labels, n_components=nc, cv=5,
                                 n_permutations=N_PERMUTATIONS,
                                 classification_method='elbow')
    dca = result['delta_ca_direct']
    dcp = result['delta_cp_direct']
    cca, cpd = result['cca'], result['cp_direct']
    print(f"\n  REPRODUCED: Delta_CA={dca:.4f}  Delta_CP={dcp:.4f}  "
          f"regime={result['regime_primary']}  "
          f"n_signal CCA={cca['n_signal']} A-SVD={cpd['n_signal']}")

    # recover per-component scores (estimator does not return them)
    phi, cca_x, cca_y = cca_scores(Z1c, Z2c)
    sigma, asvd_x = asvd_scores(Z1c, Z2c)
    assert np.allclose(phi, cca['canonical_correlations'], atol=1e-6), \
        "CCA spectrum mismatch — decomposition not aligned with estimator"
    assert np.allclose(sigma, cpd['singular_values'], atol=1e-6), \
        "A-SVD spectrum mismatch — decomposition not aligned with estimator"
    print("  score re-derivation verified against estimator spectra.")

    covariates = build_covariates(data['cov'])
    folds = _make_cv_folds(Z1c.shape[0], 5)
    cov_names = [c[0] for c in covariates]
    cov_class = {c[0]: c[1] for c in covariates}

    # per-component R^2 of score vs cell-type label (numerator/floor context)
    ct_r2_cca = cca['r2_matrix'].sum(axis=1)   # sum-R^2 used by classifier (2-D scores)
    ct_r2_asvd = cpd['r2_matrix'].sum(axis=1)

    rows = []
    for decomp, spectrum, is_signal, scores_x, scores_y, ct_r2 in [
        ('CCA',  phi,   cca['is_signal'],  cca_x,  cca_y,  ct_r2_cca),
        ('A-SVD', sigma, cpd['is_signal'], asvd_x, None,   ct_r2_asvd),
    ]:
        roles = role_map(spectrum, is_signal)
        # characterize the top-K components by singular value (the high-phi band)
        # plus any special-role component outside that band (e.g. weakest signal).
        band = list(np.argsort(spectrum)[::-1][:BAND_K])
        comps = sorted(set(int(j) for j in band) | set(roles.keys()))
        for j in comps:
            side_scores = {'rna': scores_x[:, j]}
            if scores_y is not None:
                side_scores['adt'] = scores_y[:, j]
            for side, sc in side_scores.items():
                r2map = attribute(sc, covariates, folds)
                row = {
                    'decomposition': decomp,
                    'component': j,
                    'is_signal': bool(is_signal[j]),
                    'role': roles.get(j, ''),
                    'score_side': side,
                    'singular_value': float(spectrum[j]),
                    'sumR2_vs_celltype': float(ct_r2[j]),
                }
                row.update({f'R2__{k}': r2map[k] for k in cov_names})
                rows.append(row)

    df = pd.DataFrame(rows).sort_values(
        ['decomposition', 'component', 'score_side']).reset_index(drop=True)
    csv_path = OUT_DIR / 'bmmc_floor_attribution.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # floor-setting component identity per decomposition
    floor_cca  = int(np.where(cca['is_signal'])[0].size
                     and np.where(~cca['is_signal'])[0][np.argmax(phi[~cca['is_signal']])])
    floor_asvd = int(np.where(~cpd['is_signal'])[0][np.argmax(sigma[~cpd['is_signal']])])

    # persist context for the MD writer + Exp2 gating
    meta = {
        'seed': SEED, 'max_cells': MAX_CELLS,
        'delta_ca_direct': dca, 'delta_cp_direct': dcp,
        'regime_primary': result['regime_primary'],
        'n_signal_cca': int(cca['n_signal']),
        'n_signal_asvd': int(cpd['n_signal']),
        'floor_component_cca': floor_cca,
        'floor_component_asvd': floor_asvd,
        'floor_sv_cca': float(phi[floor_cca]),
        'floor_sv_asvd': float(sigma[floor_asvd]),
        'cov_class': cov_class, 'cov_names': cov_names,
        # stored reference artifact + subsample-sensitivity context
        'stored_artifact': {'delta_ca': 0.6385, 'delta_cp': 0.5176,
                            'n_signal_cca': 16, 'n_signal_asvd': 17,
                            'floor_component_cca': 10, 'floor_sv_cca': 0.823},
        'sweep_5000_delta_ca': [0.7197, 0.3384, 0.5228, 0.4587, 0.5558],
        'sweep_5000_delta_cp': [0.5334, 0.2964, 0.3029, 0.2789, 0.2465],
    }
    with open(OUT_DIR / 'exp1_meta.json', 'w') as fh:
        json.dump(meta, fh, indent=2)

    # persist full spectra (for MD / robustness)
    np.savez(OUT_DIR / 'exp1_spectra.npz',
             cca_sv=phi, cca_is_signal=cca['is_signal'], cca_sumR2=ct_r2_cca,
             asvd_sv=sigma, asvd_is_signal=cpd['is_signal'], asvd_sumR2=ct_r2_asvd)

    # console summary table
    _print_table(df, cov_names, cov_class)
    print(f"\n  floor-setting component: CCA=comp{floor_cca} (phi={phi[floor_cca]:.3f}), "
          f"A-SVD=comp{floor_asvd} (sigma={sigma[floor_asvd]:.3f})")
    return df, meta


def _print_table(df, cov_names, cov_class):
    show = ['donor', 'batch', 'site', 'rna_log_depth', 'pct_mito',
            'adt_log_total', 'cell_cycle', 'lineage_coarse', 'cell_type']
    show = [c for c in show if c in cov_names]
    print("\n" + "=" * 128)
    print("  FLOOR ATTRIBUTION — R^2 (5-fold Ridge CV) of component score vs covariate  "
          "(sig=1 signal / 0 nuisance)")
    print("=" * 128)
    hdr = (f"  {'decomp':5} {'cmp':>3} sig {'role':16} {'side':4} {'sv':>5} {'ct':>5} | " +
           " ".join(f"{c[:8]:>8}" for c in show))
    print(hdr)
    for _, r in df.iterrows():
        line = (f"  {r['decomposition']:5} {int(r['component']):>3} "
                f"{int(bool(r['is_signal']))}   {r['role'][:16]:16} "
                f"{r['score_side']:4} {r['singular_value']:5.2f} {r['sumR2_vs_celltype']:5.2f} | " +
                " ".join(f"{r[f'R2__{c}']:8.3f}" for c in show))
        print(line)
    print("=" * 128)


# ═════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════
# Experiment 2 (conditional): re-estimate after removing donor/batch/depth
# Diagnostic-side manipulation ONLY — does NOT retrain CA/CP models.
# ═════════════════════════════════════════════════════════════════════════

def _residualize(E, D):
    """Return residuals of E after OLS regression on design D (n x p, incl. intercept)."""
    beta, *_ = np.linalg.lstsq(D, E, rcond=None)
    return E - D @ beta


def _within_group_center(E, groups):
    """Subtract per-group mean of E (removes group-level location shifts)."""
    out = E.copy()
    for g in np.unique(groups):
        m = groups == g
        out[m] -= E[m].mean(axis=0, keepdims=True)
    return out


def _estimate(Z1, Z2, labels):
    Z1c, Z2c = prereduce_and_center(Z1, Z2)
    nc = min(N_COMPONENTS, Z1c.shape[1], Z2c.shape[1])
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        res = estimate_parameters(Z1c, Z2c, labels, n_components=nc, cv=5,
                                  n_permutations=N_PERMUTATIONS,
                                  classification_method='elbow')
    return res


def _floor_of(res):
    cca, cpd = res['cca'], res['cp_direct']
    phi = cca['canonical_correlations']; sig = cpd['singular_values']
    fc = int(np.where(~cca['is_signal'])[0][np.argmax(phi[~cca['is_signal']])]) \
        if (~cca['is_signal']).any() else -1
    fa = int(np.where(~cpd['is_signal'])[0][np.argmax(sig[~cpd['is_signal']])]) \
        if (~cpd['is_signal']).any() else -1
    return {
        'delta_ca': res['delta_ca_direct'], 'delta_cp': res['delta_cp_direct'],
        'regime': res['regime_primary'],
        'n_signal_cca': int(cca['n_signal']), 'n_signal_asvd': int(cpd['n_signal']),
        'floor_cca': fc, 'floor_sv_cca': float(phi[fc]) if fc >= 0 else None,
        'floor_asvd': fa, 'floor_sv_asvd': float(sig[fa]) if fa >= 0 else None,
    }


def run_exp2():
    data = load_bmmc_subsample()
    Z1, Z2, labels, cov = data['Z1'], data['Z2'], data['labels'], data['cov']
    n = Z1.shape[0]
    intercept = np.ones((n, 1))
    log1p = lambda c: np.log1p(np.asarray(cov[c], float))

    batch_oh = _onehot(cov['batch'])            # 12 levels = site x donor (subsumes both)
    donor_oh = _onehot(cov['DonorNumber'])
    depth = np.column_stack([_zscore(log1p('ADT_total_counts')),
                             _zscore(np.log(np.asarray(cov['GEX_size_factors'], float) + 1e-8))])

    D_a = np.hstack([intercept, batch_oh, depth])   # variant (a): batch (donor+site) + depth
    variants = {
        'baseline':          (Z1, Z2),
        'regressout_batch_depth': (_residualize(Z1, D_a), _residualize(Z2, D_a)),
        'within_donor':      (_within_group_center(Z1, cov['DonorNumber'].values),
                              _within_group_center(Z2, cov['DonorNumber'].values)),
    }

    label_names = [f'ct:{c}' for c in data['ct_classes']]
    summary = {}
    for name, (z1, z2) in variants.items():
        print(f"\n=== Exp2 variant: {name} ===")
        res = _estimate(z1, z2, labels)
        f = _floor_of(res)
        summary[name] = f
        print(f"  Delta_CA={f['delta_ca']:.3f}  Delta_CP={f['delta_cp']:.3f}  "
              f"=> {f['regime']}  | nsig CCA={f['n_signal_cca']} A-SVD={f['n_signal_asvd']}  "
              f"| floor CCA=comp{f['floor_cca']}(phi={f['floor_sv_cca']}) "
              f"A-SVD=comp{f['floor_asvd']}(sig={f['floor_sv_asvd']})")
        panel = OUT_DIR / f'exp2_diagnostic_{name}.png'
        plot_diagnostic(res, 'RNA (scGPT)', 'ADT (CLR)',
                        label_names=label_names, out_path=str(panel))

    with open(OUT_DIR / 'exp2_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2)

    print("\n" + "=" * 88)
    print("  EXPERIMENT 2 — before/after removing donor/batch/depth structure")
    print("=" * 88)
    print(f"  {'variant':26} {'Delta_CA':>9} {'Delta_CP':>9} {'regime':>9} "
          f"{'nsigCCA':>8} {'floorCCA':>9}")
    for name, f in summary.items():
        print(f"  {name:26} {f['delta_ca']:9.3f} {f['delta_cp']:9.3f} {f['regime']:>9} "
              f"{f['n_signal_cca']:8d} {f['floor_sv_cca']!s:>9}")
    print("=" * 88)
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['exp1', 'exp2'], default='exp1')
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == 'exp1':
        run_exp1()
    else:
        run_exp2()


if __name__ == '__main__':
    main()
