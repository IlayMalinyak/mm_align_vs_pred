"""
Proposition 3.2 recovery count r̂ + ε×n sensitivity sweep.

For each spectrum (CCA → r̂_CA, A-SVD → r̂_CP):
    floor = max(spectrum[nuisance])
    r̂     = |{ i in signal : spectrum[i] > floor }|

Signal/nuisance classification is the elbow classifier from
src/analyze_phase_diagram.py, UNCHANGED — this script only reads off r̂ and the
associated floor / spectrum details, and re-runs the same estimator across a
grid of ε (the ridge in (Σ̂_xx+εI)^{-1/2}) and n (labeled subsample size).

Datasets:
    LAMOST × Kepler   (astro, shared label = log g)
    LAMOST × TESS     (astro, shared label = log g)
    BMMC CITE-seq     (RNA/scGPT × ADT/CLR, 44-class cell-type label)

Usage:
    python -m src.recovery_analysis                 # main table + detail + sweep
    python -m src.recovery_analysis --no-sweep      # main table + detail only
    python -m src.recovery_analysis --datasets kepler tess   # subset
"""
from __future__ import annotations
import os
import sys
import argparse
import hashlib
import numpy as np
from pathlib import Path

MM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MM_ROOT))
sys.path.insert(0, str(MM_ROOT / 'src'))
sys.path.insert(0, str(MM_ROOT / 'src' / 'bmmc'))
# MultiDESA astro loaders
MULTIDESA_ANALYSIS = os.environ.get(
    'MULTIDESA_ANALYSIS', '/rg/perets_prj/ilay.kamai/MultiDESA/analysis')
sys.path.insert(0, MULTIDESA_ANALYSIS)

# Repo estimator internals — reused verbatim (classification logic unchanged)
from src.analyze_phase_diagram import (
    _make_cv_folds, _r2_ridge_1d, _r2_ridge_2d,
    _classify_signal_elbow, recovery_count,
)

# ── Dataset configs ─────────────────────────────────────────────────────────
MULTIMODAL_DF = os.environ.get(
    'MULTIMODAL_DF', '/rg/perets_prj/ilay.kamai/multimodal/multimodal_df_min2.parquet')
_ML = '/rg/perets_prj/ilay.kamai/MultiDESA/logs'
# Kepler: the original dir 2026-02-12-07-02 was overwritten (make_kepler_catalog,
# n_matched→879, CP only). Re-extraction restores the paper regime: dir
# 2026-02-12-07-02_restore891 gives n_matched=934, regime Both (Δ̂_CA=1.078).
# The exact canonical 891 split is not reproducible (raw-LC file set drifted);
# main table uses the stored artifact, sweep uses this re-extraction. Override
# with KEPLER_LOGDIR to compare against the drifted 879 dir.
ASTRO_PAIRS = {
    'kepler': {'name': 'LAMOST × Kepler',
               'log_dir_a': f'{_ML}/lamost/2026-02-12/44644',
               'log_dir_b': os.environ.get(
                   'KEPLER_LOGDIR', f'{_ML}/kepler/2026-02-12-07-02_restore891')},
    'tess':   {'name': 'LAMOST × TESS',
               'log_dir_a': f'{_ML}/lamost/2026-02-12/44644',
               'log_dir_b': f'{_ML}/tess/2026-03-10-12-07'},
}

# canonical estimator settings (mirror the repo/MultiDESA drivers)
PCA_PREREDUCE = 100
ASTRO_NCOMP = 50
BMMC_NCOMP = 100
BMMC_MAIN_N = 5000        # canonical single-run subsample for the main table
BMMC_MAIN_SEED = 42

EPS_GRID = [1e-6, 1e-4, 1e-2, 1e-1]
N_GRID = [250, 500, 1000, 'all']
N_BOOT = 20
CV = 5


# ── Prereduce (PCA→100 + center), matching the canonical drivers ────────────
def _prereduce_and_center(Z1, Z2, k=PCA_PREREDUCE, random_state=0):
    from sklearn.decomposition import PCA
    k1 = min(k, Z1.shape[1], Z1.shape[0] - 1)
    k2 = min(k, Z2.shape[1], Z2.shape[0] - 1)
    Z1c = PCA(n_components=k1, random_state=random_state).fit_transform(Z1)
    Z2c = PCA(n_components=k2, random_state=random_state).fit_transform(Z2)
    Z1c = Z1c - Z1c.mean(axis=0)
    Z2c = Z2c - Z2c.mean(axis=0)
    return Z1c, Z2c


# ── Slim CCA / A-SVD recovery: repo math with ε exposed, elbow classifier ───
def _eigh_inv_sqrt(M):
    ev, V = np.linalg.eigh(M)
    ev = np.maximum(ev, 1e-12)
    return V @ np.diag(1.0 / np.sqrt(ev)) @ V.T


def cca_recovery(Z1c, Z2c, labels, folds, reg, n_components):
    """CCA spectrum (canonical correlations) → r̂_CA, Δ̂_CA. Matches
    _cca_analysis(classification_method='elbow') exactly, with ε=reg."""
    n = Z1c.shape[0]
    nc = min(n_components, Z1c.shape[1], Z2c.shape[1])
    Z1 = Z1c - Z1c.mean(0)
    Z2 = Z2c - Z2c.mean(0)
    Sxx = Z1.T @ Z1 / n + reg * np.eye(Z1.shape[1])
    Syy = Z2.T @ Z2 / n + reg * np.eye(Z2.shape[1])
    Sxy = Z1.T @ Z2 / n
    Sxi = _eigh_inv_sqrt(Sxx)
    Syi = _eigh_inv_sqrt(Syy)
    M = Sxi @ Sxy @ Syi
    P, phi, Qt = np.linalg.svd(M, full_matrices=False)
    nc = min(nc, len(phi))
    phi = np.clip(phi[:nc], 0, 1)
    Wx = Sxi @ P[:, :nc]
    Wy = Syi @ Qt.T[:, :nc]
    r2 = np.zeros((nc, labels.shape[1]))
    for j in range(nc):
        js = np.column_stack([Z1 @ Wx[:, j], Z2 @ Wy[:, j]])
        r2[j] = _r2_ridge_2d(js, labels, folds)
    cls = _classify_signal_elbow(r2, log_transform=False)
    is_sig = cls['is_signal']
    rec = recovery_count(phi, is_sig)
    delta = _delta(phi, is_sig)
    return phi, is_sig, rec, delta, r2.sum(axis=1), int(cls['n_signal'])


def asvd_recovery(Z1c, Z2c, labels, folds, reg, n_components):
    """A-SVD spectrum of A = Σ_yx Σ_xx^{-1/2} → r̂_CP, Δ̂_CP. Matches
    _cp_direct_analysis(classification_method='elbow') exactly, with ε=reg."""
    n = Z1c.shape[0]
    nc = min(n_components, Z1c.shape[1], Z2c.shape[1])
    Z1 = Z1c - Z1c.mean(0)
    Z2 = Z2c - Z2c.mean(0)
    Sxx = Z1.T @ Z1 / n + reg * np.eye(Z1.shape[1])
    Syx = Z2.T @ Z1 / n
    Sxi = _eigh_inv_sqrt(Sxx)
    A = Syx @ Sxi
    U_a, sigma_a, Vt_a = np.linalg.svd(A, full_matrices=False)
    V_a = Vt_a.T
    nc = min(nc, len(sigma_a))
    sigma_a = sigma_a[:nc]
    V_a = V_a[:, :nc]
    W_src = Sxi @ V_a
    r2 = np.zeros((nc, labels.shape[1]))
    for j in range(nc):
        r2[j] = _r2_ridge_1d(Z1 @ W_src[:, j], labels, folds)
    cls = _classify_signal_elbow(r2, log_transform=False)
    is_sig = cls['is_signal']
    rec = recovery_count(sigma_a, is_sig)
    delta = _delta(sigma_a, is_sig)
    return sigma_a, is_sig, rec, delta, r2.sum(axis=1), int(cls['n_signal'])


def _delta(spectrum, is_sig):
    is_nui = ~is_sig
    if is_sig.any() and is_nui.any():
        return float(spectrum[is_sig].min() / max(spectrum[is_nui].max(), 1e-15))
    if is_sig.any():
        return float('inf')
    return 0.0


def _regime(dca, dcp):
    if dca > 1 and dcp > 1:
        return 'Both'
    if dca > 1:
        return 'CA only'
    if dcp > 1:
        return 'CP only'
    return 'Neither'


def estimate_cell(Z1c, Z2c, labels, reg, n_components):
    """One estimation: returns dict of spectra/is_signal/recovery/deltas."""
    folds = _make_cv_folds(Z1c.shape[0], cv=CV)
    phi, sig_ca, rec_ca, dca, cca_sum, cca_bp = cca_recovery(Z1c, Z2c, labels, folds, reg, n_components)
    sig_a, sig_cp, rec_cp, dcp, asvd_sum, asvd_bp = asvd_recovery(Z1c, Z2c, labels, folds, reg, n_components)
    return {
        'cca_spectrum': phi, 'cca_is_signal': sig_ca, 'cca_rec': rec_ca, 'delta_ca': dca,
        'asvd_spectrum': sig_a, 'asvd_is_signal': sig_cp, 'asvd_rec': rec_cp, 'delta_cp': dcp,
        'regime': _regime(dca, dcp),
        'cca_sum_r2': cca_sum, 'cca_breakpoint': cca_bp,
        'asvd_sum_r2': asvd_sum, 'asvd_breakpoint': asvd_bp,
    }


# ── Loaders: return RAW (pre-prereduce) arrays so we can resample then reduce ─
def load_astro(pair_key):
    import pandas as pd
    from analyze_phase_diagram import (
        load_data, match_samples, prepare_labels_matrix, SHARED_LABEL_KEYS)
    cfg = ASTRO_PAIRS[pair_key]
    Ua, dfa = load_data(cfg['log_dir_a'], mode='test')
    Ub, dfb = load_data(cfg['log_dir_b'], mode='test')
    mm = pd.read_parquet(MULTIMODAL_DF)
    ia, ib, labels_df = match_samples(dfa, dfb, mm)
    X = np.asarray(Ua[ia], dtype=np.float64)
    Y = np.asarray(Ub[ib], dtype=np.float64)
    del Ua, Ub
    shared, _ = prepare_labels_matrix(labels_df, keep_keys=SHARED_LABEL_KEYS)
    return {'name': cfg['name'], 'X': X, 'Y': Y, 'labels': np.asarray(shared, float),
            'n_components': ASTRO_NCOMP, 'raw_dx': X.shape[1], 'raw_dy': Y.shape[1],
            'label_desc': 'log g (shared)'}


BMMC_RNA_EMB = os.environ.get(
    'BMMC_RNA_EMB', '/home/ilay.kamai/work/citeseq/bmmc/rna_embeddings_scgpt.npy')
BMMC_H5AD = os.environ.get(
    'BMMC_H5AD', '/home/ilay.kamai/work/citeseq/bmmc/neurips2021_cite_BMMC.h5ad')


def _one_hot(values, classes):
    c2i = {c: i for i, c in enumerate(classes)}
    arr = np.zeros((len(values), len(classes)), dtype=np.float32)
    for row, v in enumerate(values):
        arr[row, c2i[v]] = 1.0
    return arr


def load_bmmc():
    """Inline of src/bmmc/analyze_phase_diagram_bmmc.load_data (raw CLR ADT).
    Z1 = scGPT RNA embeddings, Z2 = raw CLR ADT, labels = cell_type one-hot."""
    import anndata
    import scipy.sparse as sp
    from bmmc_dataset import load_bmmc as _load_bmmc
    adata = anndata.read_h5ad(BMMC_H5AD)
    Z1_full = np.load(BMMC_RNA_EMB).astype(np.float32)
    _, adt_adata, obs_df = _load_bmmc(BMMC_H5AD)
    X_adt = adt_adata.X
    if sp.issparse(X_adt):
        X_adt = X_adt.toarray()
    X_adt = X_adt.astype(np.float32)
    n_full, n_filt = adata.shape[0], len(obs_df)
    if Z1_full.shape[0] == n_full:
        int_idx = __import__('pandas').Index(adata.obs_names).get_indexer(obs_df.index)
        assert (int_idx >= 0).all()
        Z1_full = Z1_full[int_idx]
    elif Z1_full.shape[0] != n_filt:
        raise ValueError(f"Z1 shape mismatch: {Z1_full.shape[0]} vs {n_full}/{n_filt}")
    assert Z1_full.shape[0] == X_adt.shape[0] == n_filt
    cell_types = obs_df['cell_type'].values
    ct_classes = sorted(set(cell_types))
    labels = _one_hot(cell_types, ct_classes)
    return {'name': 'BMMC (RNA/scGPT × ADT/CLR)', 'X': np.asarray(Z1_full, float),
            'Y': np.asarray(X_adt, float), 'labels': np.asarray(labels, float),
            'n_components': BMMC_NCOMP, 'raw_dx': Z1_full.shape[1], 'raw_dy': X_adt.shape[1],
            'label_desc': f'cell_type ({labels.shape[1]} classes)'}


# ── Astro main-table/detail from the CANONICAL stored artifact ──────────────
# The Kepler encoder outputs at logs/kepler/2026-02-12-07-02 were overwritten
# (features 2026-05-06, preds 2026-05-13) AFTER the Fig-12 artifact (2026-04-23),
# shifting the test split (n_matched 891→879) and the CCA spectrum. A fresh run
# on current files therefore diverges from the paper. The stored classification
# npz holds the exact Fig-12 spectra + is_signal, so astro r̂ is read from there.
ASTRO_NPZ_DIR = os.environ.get(
    'ASTRO_NPZ_DIR',
    '/rg/perets_prj/ilay.kamai/MultiDESA/analysis/multimodal_comparison')
ASTRO_ARTIFACT = {'kepler': 'phase_lamost_x_kepler', 'tess': 'phase_lamost_x_tess'}


def load_astro_artifact(pair_key):
    """Read canonical CCA + A-SVD spectra/is_signal from the stored npz/json."""
    import json
    stem = ASTRO_ARTIFACT[pair_key]
    d = np.load(os.path.join(ASTRO_NPZ_DIR, f'{stem}_classification_data.npz'),
                allow_pickle=True)
    j = json.load(open(os.path.join(ASTRO_NPZ_DIR, f'{stem}_result.json')))
    cca_sv = d['cca_sv'].astype(float)
    cca_sig = d['cca_is_signal'].astype(bool)
    cp_sv = d['cp_direct_sv'].astype(float)
    cp_sig = d['cp_direct_is_signal'].astype(bool)
    name = 'LAMOST × Kepler' if pair_key == 'kepler' else 'LAMOST × TESS'
    r = {
        'cca_spectrum': cca_sv, 'cca_is_signal': cca_sig,
        'cca_rec': recovery_count(cca_sv, cca_sig), 'delta_ca': _delta(cca_sv, cca_sig),
        'asvd_spectrum': cp_sv, 'asvd_is_signal': cp_sig,
        'asvd_rec': recovery_count(cp_sv, cp_sig), 'delta_cp': _delta(cp_sv, cp_sig),
    }
    r['regime'] = _regime(r['delta_ca'], r['delta_cp'])
    ds = {'name': name, 'n_components': int(j.get('n_components', 50)),
          'raw_dx': 2048, 'raw_dy': 512, 'label_desc': 'log g (shared)'}
    return ds, r, int(j['n_matched'])


def astro_artifact_row_detail(pair_key):
    ds, r, n = load_astro_artifact(pair_key)
    dx = dy = PCA_PREREDUCE   # canonical pca_prereduce=100 → Σ̂ dimension
    rca, rcp = r['cca_rec'], r['asvd_rec']
    row = {'name': ds['name'], 'n': n, 'dx': dx, 'dy': dy, 'eps': 1e-6,
           'k_ca': rca['k_hat'], 'r_ca': rca['r_hat'], 'dca': r['delta_ca'],
           'k_cp': rcp['k_hat'], 'r_cp': rcp['r_hat'], 'dcp': r['delta_cp'],
           'regime': r['regime']}
    detail = (ds, n, None, dx, dy, 1e-6, r)
    return row, detail


# ── Main-table estimate (canonical subsample, ε=1e-6) ───────────────────────
def canonical_subsample(ds):
    """Return (X, Y, labels, n_used, seed) for the main-table single run."""
    N = ds['X'].shape[0]
    if ds['name'].startswith('BMMC'):
        rng = np.random.default_rng(BMMC_MAIN_SEED)
        idx = np.sort(rng.choice(N, BMMC_MAIN_N, replace=False))
        return ds['X'][idx], ds['Y'][idx], ds['labels'][idx], BMMC_MAIN_N, BMMC_MAIN_SEED
    return ds['X'], ds['Y'], ds['labels'], N, None   # astro: all matched pairs


def fmt_delta(d):
    if d == float('inf'):
        return '  inf'
    return f'{d:6.3f}'


def run_main(datasets, reg=1e-6):
    rows = []
    details = []
    for ds in datasets:
        X, Y, lab, n_used, seed = canonical_subsample(ds)
        Z1c, Z2c = _prereduce_and_center(X, Y)
        dx, dy = Z1c.shape[1], Z2c.shape[1]
        r = estimate_cell(Z1c, Z2c, lab, reg, ds['n_components'])
        rca, rcp = r['cca_rec'], r['asvd_rec']
        rows.append({
            'name': ds['name'], 'n': n_used, 'dx': dx, 'dy': dy, 'eps': reg,
            'k_ca': rca['k_hat'], 'r_ca': rca['r_hat'], 'dca': r['delta_ca'],
            'k_cp': rcp['k_hat'], 'r_cp': rcp['r_hat'], 'dcp': r['delta_cp'],
            'regime': r['regime'],
        })
        details.append((ds, n_used, seed, dx, dy, reg, r))
    return rows, details


def print_main_table(rows):
    print("\n" + "=" * 118)
    print("MAIN TABLE — Proposition 3.2 recovery count r̂  (ε = 1e-6, elbow classifier)")
    print("=" * 118)
    hdr = (f"{'dataset':<28}{'n':>7}{'d_x':>5}{'d_y':>5}"
           f"{'k̂_CA':>6}{'r̂_CA':>6}{'Δ̂_CA':>8}"
           f"{'k̂_CP':>6}{'r̂_CP':>6}{'Δ̂_CP':>8}{'regime':>10}")
    print(hdr)
    print("-" * 118)
    for x in rows:
        print(f"{x['name']:<28}{x['n']:>7}{x['dx']:>5}{x['dy']:>5}"
              f"{x['k_ca']:>6}{x['r_ca']:>6}{fmt_delta(x['dca']):>8}"
              f"{x['k_cp']:>6}{x['r_cp']:>6}{fmt_delta(x['dcp']):>8}{x['regime']:>10}")
    print("=" * 118)


def print_details(details):
    for ds, n_used, seed, dx, dy, reg, r in details:
        print("\n" + "─" * 100)
        print(f"DETAIL — {ds['name']}")
        print(f"  label = {ds['label_desc']} | ε = {reg:g} | n = {n_used}"
              + (f" (seed {seed})" if seed is not None else " (all matched)")
              + f" | raw d_x = {ds['raw_dx']}, raw d_y = {ds['raw_dy']}"
              f" | prereduced d_x = {dx}, d_y = {dy} | n_components = {ds['n_components']}")
        for tag, spec_key, sig_key, rec_key, dkey in [
            ('CCA  (r̂_CA, Δ̂_CA)', 'cca_spectrum', 'cca_is_signal', 'cca_rec', 'delta_ca'),
            ('A-SVD(r̂_CP, Δ̂_CP)', 'asvd_spectrum', 'asvd_is_signal', 'asvd_rec', 'delta_cp')]:
            rec = r[rec_key]
            print(f"  {tag}:  k̂={rec['k_hat']}  r̂={rec['r_hat']}  "
                  f"Δ̂={fmt_delta(r[dkey]).strip()}  "
                  f"floor={rec['floor']:.5f} @ component index {rec['floor_index']}")
            pairs = ', '.join(f'[{int(i)}]{v:.5f}'
                              for i, v in zip(rec['signal_indices'], rec['signal_values']))
            print(f"        signal spectrum (idx↓value, desc): {pairs}")
            above = int(np.sum(rec['signal_values'] > rec['floor']))
            contig = list(map(int, sorted(rec['signal_indices'])))
            print(f"        signal indices sorted: {contig}  "
                  f"(contiguous block 0..{rec['k_hat']-1}? "
                  f"{contig == list(range(rec['k_hat']))})  |  #signal>floor = {above}")


# ── Sensitivity sweep ───────────────────────────────────────────────────────
def _iqr(a):
    a = np.asarray(a, float)
    return float(np.percentile(a, 25)), float(np.percentile(a, 75))


def run_sweep(datasets):
    print("\n\n" + "=" * 120)
    print(f"SENSITIVITY SWEEP — ε × n, {N_BOOT} bootstrap resamples/cell "
          f"(median [IQR]); baseline = ε=1e-6 at same n")
    print("=" * 120)
    for ds in datasets:
        N = ds['X'].shape[0]
        n_list = [n for n in N_GRID if n == 'all' or n <= N]
        skipped = [n for n in N_GRID if n != 'all' and n > N]
        print(f"\n### {ds['name']}   (N_available = {N}, n_components = {ds['n_components']})"
              + (f"   [skipped n>{N}: {skipped}]" if skipped else ""))
        hdr = (f"{'n':>6}{'ε':>8}"
               f"{'r̂_CA med[IQR]':>18}{'r̂_CP med[IQR]':>18}"
               f"{'Δ̂_CA med[IQR]':>22}{'Δ̂_CP med[IQR]':>22}"
               f"{'regime(med)':>12}{'flag':>6}")
        print(hdr)
        print("-" * 120)
        for n_spec in n_list:
            n_use = N if n_spec == 'all' else n_spec
            # collect per-bootstrap resamples ONCE, evaluate all ε on each
            boot_idx = []
            boot_reduced = []
            name_seed = int(hashlib.md5(ds['name'].encode()).hexdigest()[:8], 16) % 9973
            for b in range(N_BOOT):
                rng = np.random.default_rng(1000 * name_seed + 7 * n_use + b)
                idx = rng.choice(N, n_use, replace=True)
                Z1c, Z2c = _prereduce_and_center(ds['X'][idx], ds['Y'][idx])
                boot_reduced.append((Z1c, Z2c, ds['labels'][idx]))
            cell_stats = {}
            for reg in EPS_GRID:
                rca, rcp, dca, dcp, regs = [], [], [], [], []
                for (Z1c, Z2c, lab) in boot_reduced:
                    r = estimate_cell(Z1c, Z2c, lab, reg, ds['n_components'])
                    rca.append(r['cca_rec']['r_hat'])
                    rcp.append(r['asvd_rec']['r_hat'])
                    dca.append(r['delta_ca'] if np.isfinite(r['delta_ca']) else np.nan)
                    dcp.append(r['delta_cp'] if np.isfinite(r['delta_cp']) else np.nan)
                    regs.append(r['regime'])
                cell_stats[reg] = {
                    'rca': rca, 'rcp': rcp, 'dca': dca, 'dcp': dcp, 'regs': regs}
            # baseline = eps 1e-6
            base = cell_stats[1e-6]
            base_rca_med = np.median(base['rca'])
            base_rcp_med = np.median(base['rcp'])
            base_reg = _regime(np.nanmedian(base['dca']), np.nanmedian(base['dcp']))
            for reg in EPS_GRID:
                s = cell_stats[reg]
                rca_med = np.median(s['rca']); rcp_med = np.median(s['rcp'])
                dca_med = np.nanmedian(s['dca']); dcp_med = np.nanmedian(s['dcp'])
                reg_med = _regime(dca_med, dcp_med)
                flags = ''
                if rca_med != base_rca_med or rcp_med != base_rcp_med:
                    flags += 'r'
                if reg_med != base_reg:
                    flags += 'R'
                # within-cell regime instability: does Δ IQR straddle 1?
                for arr in (s['dca'], s['dcp']):
                    lo, hi = _iqr(np.asarray(arr, float)[~np.isnan(arr)]) if np.any(~np.isnan(arr)) else (np.nan, np.nan)
                    if np.isfinite(lo) and np.isfinite(hi) and lo < 1.0 < hi:
                        flags += '±'
                        break
                rca_i = _iqr(s['rca']); rcp_i = _iqr(s['rcp'])
                dca_i = _iqr(np.asarray(s['dca'])[~np.isnan(s['dca'])]) if np.any(~np.isnan(s['dca'])) else (np.nan, np.nan)
                dcp_i = _iqr(np.asarray(s['dcp'])[~np.isnan(s['dcp'])]) if np.any(~np.isnan(s['dcp'])) else (np.nan, np.nan)
                print(f"{n_use:>6}{reg:>8.0e}"
                      f"{f'{rca_med:.0f} [{rca_i[0]:.0f},{rca_i[1]:.0f}]':>18}"
                      f"{f'{rcp_med:.0f} [{rcp_i[0]:.0f},{rcp_i[1]:.0f}]':>18}"
                      f"{f'{dca_med:.2f} [{dca_i[0]:.2f},{dca_i[1]:.2f}]':>22}"
                      f"{f'{dcp_med:.2f} [{dcp_i[0]:.2f},{dcp_i[1]:.2f}]':>22}"
                      f"{reg_med:>12}{flags:>6}")
        print("  flags: r=median r̂ differs from ε=1e-6 baseline | "
              "R=regime(median Δ̂) differs from baseline | ±=Δ̂ IQR straddles 1")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='*',
                   default=['kepler', 'tess', 'bmmc'],
                   choices=['kepler', 'tess', 'bmmc'])
    p.add_argument('--no-sweep', action='store_true')
    p.add_argument('--astro-fresh', action='store_true',
                   help='Estimate astro from current (re-extracted) files instead of '
                        'the canonical stored artifact. Diverges from the paper (879 vs 891).')
    return p.parse_args()


def main():
    args = parse_args()
    rows, details = [], []
    sweep_datasets = []
    astro_from_artifact = []
    for key in args.datasets:
        print(f"\n[load] {key} ...", flush=True)
        if key in ('kepler', 'tess') and not args.astro_fresh:
            row, detail = astro_artifact_row_detail(key)   # canonical stored spectra
            rows.append(row); details.append(detail)
            astro_from_artifact.append(row['name'])
            if not args.no_sweep:
                sweep_datasets.append(load_astro(key))      # fresh, for ε×n sweep only
        else:
            ds = load_astro(key) if key in ('kepler', 'tess') else load_bmmc()
            r_rows, r_details = run_main([ds])
            rows += r_rows; details += r_details
            sweep_datasets.append(ds)
    print_main_table(rows)
    if astro_from_artifact:
        print(f"  NOTE: {', '.join(astro_from_artifact)} rows read from the canonical "
              f"stored classification npz (Fig-12 artifact, n_matched 891/153).")
        print(f"        BMMC is a fresh run. Astro fresh re-runs (879, drifted inputs) "
              f"available via --astro-fresh.")
    print_details(details)
    if not args.no_sweep:
        if astro_from_artifact:
            print("\n  NOTE: the ε×n sweep below re-runs the estimator on CURRENT astro "
                  "files (re-extracted Kepler dir 2026-02-12-07-02_restore891),")
            print("        distinct from the canonical stored artifact used in the main "
                  "table above. BMMC sweep is on unchanged inputs.")
        run_sweep(sweep_datasets)


if __name__ == '__main__':
    main()
