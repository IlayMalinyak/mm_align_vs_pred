"""
Label-quality robustness study for the recovery-regime diagnostic (Algorithm 1,
elbow classifier in src/analyze_phase_diagram.py).

The estimator is UNCHANGED. It uses labels only to classify each spectrum
component as signal vs nuisance via 5-fold Ridge CV R². This script re-runs the
SAME estimator at a fixed ε=1e-6, varying ONLY the label matrix, to see how r̂
and the regime call degrade as label quality drops.

Label variants (one estimator run per variant per dataset):
  (0) ground-truth        astro = log g;   BMMC = 44-class cell_type one-hot
  (1) label noise         astro = log g + Gaussian noise calibrated so predictive
                          R²(label vs features) ≈ {0.75,0.50,0.25}× baseline
                          BMMC  = reassign a fraction f∈{0.1,0.25,0.5} of cells to
                          a uniformly random class
  (2) coarsened labels    astro = log g quantized into {8,4} quantile bins (one-hot)
                          BMMC  = 44 cell types folded to ~8 lineages (one-hot)
  (3) unsupervised proxy  one-hot of k-means cluster ids computed on the WEAKER
                          modality's raw features (astro: Kepler/TESS photometry Y;
                          BMMC: ADT Y), n_clusters∈{8,20,44}

Datasets: LAMOST×Kepler (restored 934 dir — the stored 891 artifact holds only the
classification spectra, not the raw features/labels needed to re-run with new
labels), LAMOST×TESS (153), BMMC (n=5000, seed 42; same subsample as the main table).

Markdown tables only, ε=1e-6 fixed throughout. No interpretation.

    python -m src.label_robustness --datasets kepler tess bmmc
"""
from __future__ import annotations
import argparse
import numpy as np

from src import recovery_analysis as RA
from src.recovery_analysis import (
    load_astro, _prereduce_and_center, estimate_cell,
    BMMC_MAIN_N, BMMC_MAIN_SEED)
from src.analyze_phase_diagram import _make_cv_folds

EPS = 1e-6
KM_SEED = 0
KM_NINIT = 10
NOISE_SEED = 0
KM_CLUSTERS = [8, 20, 44]
NOISE_TARGET_RATIOS = [0.75, 0.50, 0.25]
BMMC_REASSIGN_FRAC = [0.10, 0.25, 0.50]
ASTRO_BINS = [8, 4]
CV = 5


# ── label helpers ────────────────────────────────────────────────────────────
def _one_hot_int(idx, k=None):
    idx = np.asarray(idx).astype(int)
    k = int(idx.max()) + 1 if k is None else int(k)
    A = np.zeros((idx.size, k), dtype=float)
    A[np.arange(idx.size), idx] = 1.0
    return A


def _one_hot_str(values, classes):
    c2i = {c: i for i, c in enumerate(classes)}
    A = np.zeros((len(values), len(classes)), dtype=float)
    for r, v in enumerate(values):
        A[r, c2i[v]] = 1.0
    return A


def _cv_r2_multi(F, y, folds, alpha=1.0):
    """Pooled 5-fold Ridge R² predicting 1-D y from many features F.
    Calibration knob only (not a reported estimator quantity)."""
    from sklearn.linear_model import Ridge
    n = len(y)
    ss_res = ss_tot = 0.0
    mask = np.ones(n, dtype=bool)
    for te in folds:
        mask[:] = True
        mask[te] = False
        m = Ridge(alpha=alpha).fit(F[mask], y[mask])
        p = m.predict(F[te])
        ss_res += np.sum((y[te] - p) ** 2)
        ss_tot += np.sum((y[te] - y[te].mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0


def _calibrate_noise(logg, e, F, folds, baseline, ratio, iters=40, tol=0.004):
    """Find noise std σ so that R²(logg+σ·e vs F)/baseline ≈ ratio (e fixed)."""
    sd = float(logg.std())
    lo, hi = 0.0, 10.0 * sd + 1e-9
    for _ in range(40):                       # grow hi until ratio undershoots
        if _cv_r2_multi(F, logg + hi * e, folds) / baseline < ratio:
            break
        hi *= 1.5
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        rm = _cv_r2_multi(F, logg + mid * e, folds) / baseline
        if abs(rm - ratio) < tol:
            return mid, rm
        if rm > ratio:                         # too much signal left → more noise
            lo = mid
        else:
            hi = mid
    mid = 0.5 * (lo + hi)
    return mid, _cv_r2_multi(F, logg + mid * e, folds) / baseline


# ── BMMC full loader (returns cell_type STRINGS; mirrors RA.load_bmmc) ────────
def load_bmmc_full():
    import anndata
    import scipy.sparse as sp
    import pandas as pd
    from bmmc_dataset import load_bmmc as _load_bmmc
    adata = anndata.read_h5ad(RA.BMMC_H5AD)
    Z1_full = np.load(RA.BMMC_RNA_EMB).astype(np.float32)
    _, adt_adata, obs_df = _load_bmmc(RA.BMMC_H5AD)
    X_adt = adt_adata.X
    if sp.issparse(X_adt):
        X_adt = X_adt.toarray()
    X_adt = X_adt.astype(np.float32)
    n_full, n_filt = adata.shape[0], len(obs_df)
    if Z1_full.shape[0] == n_full:
        int_idx = pd.Index(adata.obs_names).get_indexer(obs_df.index)
        assert (int_idx >= 0).all()
        Z1_full = Z1_full[int_idx]
    elif Z1_full.shape[0] != n_filt:
        raise ValueError(f"Z1 shape mismatch {Z1_full.shape[0]} vs {n_full}/{n_filt}")
    cell_types = np.asarray(obs_df['cell_type'].values).astype(str)
    return np.asarray(Z1_full, float), np.asarray(X_adt, float), cell_types


def bmmc_lineage(name):
    """Fold a BMMC/CITE-seq cell-type name to a coarse lineage. Rule-based on the
    standard neurips2021 naming; the exact resulting grouping is printed in
    provenance so the mapping is fully specified."""
    n = name.lower()
    if 'prog' in n or n.startswith('hsc') or 'proerythroblast' in n or \
       'erythroblast' in n or 'normoblast' in n or 'reticulocyte' in n or \
       'megakaryocyte' in n:
        return 'Progenitor/Erythroid'
    if 'plasma' in n:                         # plasma cell / plasmablast
        return 'Plasma'
    if n.startswith('b1 b') or n == 'b cells' or ' b ' in f' {n} ' or \
       'transitional b' in n or 'naive cd20' in n or 'memory b' in n or \
       n.endswith(' b') or 'igkc' in n:
        return 'B'
    if 'cd8' in n and 't' in n:
        return 'CD8 T'
    if 'cd4' in n and 't' in n:
        return 'CD4 T'
    if 'treg' in n or 't reg' in n or 'mait' in n or 'gdt' in n or 'dnt' in n or \
       'ilc' in n or n == 't cells' or n.endswith(' t'):
        return 'other T/innate-lymphoid'
    if n.startswith('nk') or ' nk' in n:
        return 'NK'
    if 'mono' in n or 'dc' in n:             # monocytes + dendritic
        return 'Myeloid'
    return 'other'


# ── build variant label matrices for one dataset ─────────────────────────────
def astro_variants(ds, Z1c, Z2c, folds):
    """Return list of (variant_name, detail_str, labels_matrix) for astro."""
    logg = np.asarray(ds['labels'], float)
    if logg.ndim == 2:
        assert logg.shape[1] == 1, f"astro label expected 1 col, got {logg.shape}"
        logg = logg[:, 0]
    out = []
    # (0) ground truth
    out.append(('ground-truth (log g)', 'L=1', logg.reshape(-1, 1)))
    # (1) label noise
    e = np.random.default_rng(NOISE_SEED).standard_normal(logg.size)
    F = np.column_stack([Z1c, Z2c])
    baseline = _cv_r2_multi(F, logg, folds)
    for ratio in NOISE_TARGET_RATIOS:
        sigma, ach = _calibrate_noise(logg, e, F, folds, baseline, ratio)
        det = f'σ={sigma:.4f} (={sigma/logg.std():.3f}·std); R²ratio target {ratio:.2f} got {ach:.3f}'
        out.append((f'noise R²≈{ratio:.2f}× base', det, (logg + sigma * e).reshape(-1, 1)))
    # (2) coarsened — quantile bins
    for B in ASTRO_BINS:
        edges = np.quantile(logg, np.linspace(0, 1, B + 1))
        bins = np.clip(np.digitize(logg, edges[1:-1]), 0, B - 1)
        out.append((f'quantize {B} bins', f'L={B} quantile bins', _one_hot_int(bins, B)))
    # (3) unsupervised k-means on weaker modality (Y = photometry, raw)
    from sklearn.cluster import KMeans
    Yraw = np.asarray(ds['Y'], float)
    for k in KM_CLUSTERS:
        cl = KMeans(n_clusters=k, random_state=KM_SEED, n_init=KM_NINIT).fit_predict(Yraw)
        out.append((f'k-means k={k} (Y)', f'L={k}; on raw Y d={Yraw.shape[1]}',
                    _one_hot_int(cl, k)))
    return out, baseline


def bmmc_variants(Z1c, Z2c, Yraw_sub, cell_types_sub, ct_classes):
    """Return list of (variant_name, detail_str, labels_matrix) for BMMC."""
    out = []
    n = len(cell_types_sub)
    # (0) ground truth
    out.append((f'ground-truth ({len(ct_classes)} cell types)', f'L={len(ct_classes)}',
                _one_hot_str(cell_types_sub, ct_classes)))
    # (1) random reassignment of a fraction f
    for f in BMMC_REASSIGN_FRAC:
        rng = np.random.default_rng(NOISE_SEED)
        ct = cell_types_sub.copy()
        m = int(round(f * n))
        pos = rng.choice(n, m, replace=False)
        ct[pos] = rng.choice(ct_classes, m, replace=True)
        out.append((f'reassign f={f:.2f}', f'{m}/{n} cells randomized',
                    _one_hot_str(ct, ct_classes)))
    # (2) coarsened lineages
    lin = np.asarray([bmmc_lineage(c) for c in cell_types_sub])
    lin_classes = sorted(set(lin))
    out.append((f'coarse lineages ({len(lin_classes)})', f'L={len(lin_classes)}',
                _one_hot_str(lin, lin_classes)))
    # (3) unsupervised k-means on ADT (weaker modality Y, raw)
    from sklearn.cluster import KMeans
    for k in KM_CLUSTERS:
        cl = KMeans(n_clusters=k, random_state=KM_SEED, n_init=KM_NINIT).fit_predict(Yraw_sub)
        out.append((f'k-means k={k} (ADT)', f'L={k}; on raw ADT d={Yraw_sub.shape[1]}',
                    _one_hot_int(cl, k)))
    return out, (cell_types_sub, ct_classes)


# ── run + print ──────────────────────────────────────────────────────────────
def _fmt_d(d):
    return 'inf' if d == float('inf') else f'{d:.3f}'


def run_dataset(name, Z1c, Z2c, variants, n_components, n_used):
    folds = None
    rows = []
    for vname, det, lab in variants:
        r = estimate_cell(Z1c, Z2c, lab, EPS, n_components)
        rca, rcp = r['cca_rec'], r['asvd_rec']
        rows.append({
            'variant': vname, 'detail': det,
            'k_ca': rca['k_hat'], 'r_ca': rca['r_hat'], 'd_ca': r['delta_ca'],
            'bp_ca': r['cca_breakpoint'],
            'k_cp': rcp['k_hat'], 'r_cp': rcp['r_hat'], 'd_cp': r['delta_cp'],
            'bp_cp': r['asvd_breakpoint'], 'regime': r['regime'],
        })
    print(f"\n### {name}  (ε=1e-6, n={n_used}, n_components={n_components})\n")
    print("| variant | k̂_CA | r̂_CA | Δ̂_CA | bp_CA | k̂_CP | r̂_CP | Δ̂_CP | bp_CP | regime | detail |")
    print("|--|--:|--:|--:|--:|--:|--:|--:|--:|--|--|")
    for x in rows:
        print(f"| {x['variant']} | {x['k_ca']} | {x['r_ca']} | {_fmt_d(x['d_ca'])} | {x['bp_ca']} | "
              f"{x['k_cp']} | {x['r_cp']} | {_fmt_d(x['d_cp'])} | {x['bp_cp']} | {x['regime']} | {x['detail']} |")
    # summary vs baseline (row 0)
    base = rows[0]
    print(f"\n#### {name} — agreement with ground-truth baseline\n")
    print("| variant | regime matches baseline? | r̂_CA sign (0 vs >0) matches? |")
    print("|--|--|--|")
    for x in rows:
        reg_ok = 'Yes' if x['regime'] == base['regime'] else 'No'
        sign_ok = 'Yes' if (x['r_ca'] > 0) == (base['r_ca'] > 0) else 'No'
        tag = ' _(baseline)_' if x is base else ''
        print(f"| {x['variant']}{tag} | {reg_ok} | {sign_ok} |")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', nargs='*', default=['kepler', 'tess', 'bmmc'],
                   choices=['kepler', 'tess', 'bmmc'])
    args = p.parse_args()

    import numpy as _np
    import scipy
    import sklearn
    print("# Label-quality robustness of the recovery-regime diagnostic (ε=1e-6)\n")
    print("Estimator unchanged (elbow classifier); only the label matrix varies. "
          "Every cell is a single deterministic estimator run at ε=1e-6.\n")

    prov_noise = {}
    prov_lineage = None

    for key in args.datasets:
        if key in ('kepler', 'tess'):
            ds = load_astro(key)
            X, Y, lab, n_used = ds['X'], ds['Y'], ds['labels'], ds['X'].shape[0]
            Z1c, Z2c = _prereduce_and_center(X, Y)
            folds = _make_cv_folds(Z1c.shape[0], cv=CV)
            variants, baseline = astro_variants(ds, Z1c, Z2c, folds)
            prov_noise[ds['name']] = (baseline, [(v, d) for v, d, _ in variants
                                                 if v.startswith('noise')])
            run_dataset(ds['name'], Z1c, Z2c, variants, ds['n_components'], n_used)
        else:
            Z1f, Yf, ctf = load_bmmc_full()
            N = Z1f.shape[0]
            rng = _np.random.default_rng(BMMC_MAIN_SEED)
            idx = _np.sort(rng.choice(N, BMMC_MAIN_N, replace=False))
            Xs, Ys, cts = Z1f[idx], Yf[idx], ctf[idx]
            ct_classes = sorted(set(ctf))     # full 44-class order (matches load_bmmc)
            Z1c, Z2c = _prereduce_and_center(Xs, Ys)
            variants, (cts_sub, _) = bmmc_variants(Z1c, Z2c, Ys, cts, ct_classes)
            # capture lineage mapping for provenance
            lin = {c: bmmc_lineage(c) for c in ct_classes}
            prov_lineage = lin
            run_dataset('BMMC (RNA/scGPT × ADT/CLR)', Z1c, Z2c, variants,
                        RA.BMMC_NCOMP, BMMC_MAIN_N)

    # ── provenance ───────────────────────────────────────────────────────────
    print("\n## Provenance\n")
    print("| field | value |")
    print("|---|---|")
    print(f"| ε (fixed) | 1e-6 |")
    print(f"| Estimator | `estimate_cell` → `src/analyze_phase_diagram.py` elbow classifier, unchanged |")
    print(f"| CV | 5-fold, `_make_cv_folds` seed 42; Ridge α=1.0 |")
    print(f"| Prereduce | PCA→100 + center (`random_state=0`), applied once per dataset |")
    print(f"| Astro Kepler dir | restored 934 (`..._restore891`); 891 artifact lacks raw features/labels |")
    print(f"| k-means | `sklearn.cluster.KMeans`, `random_state={KM_SEED}`, `n_init={KM_NINIT}`, on RAW weaker modality Y |")
    print(f"| Astro noise | logg + σ·e, e~N(0,1) fixed seed {NOISE_SEED}; σ bisection-calibrated to target R² ratio; calibration R² = pooled 5-fold Ridge(α=1) of label on [Z1c|Z2c] |")
    print(f"| BMMC reassign | fraction f∈{{0.10,0.25,0.50}} of cells → uniform random class among the 44; seed {NOISE_SEED} |")
    print(f"| Astro quantize | quantile bins (`np.quantile`, equal-frequency), B∈{{8,4}} |")
    print(f"| Libraries | numpy {_np.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__} |")

    for nm, (base, noises) in prov_noise.items():
        print(f"\n**{nm} — noise calibration** (baseline R²={base:.4f}):\n")
        print("| target ratio | detail |")
        print("|--|--|")
        for v, d in noises:
            print(f"| {v} | {d} |")

    if prov_lineage is not None:
        print("\n**BMMC 44→lineage mapping used:**\n")
        groups = {}
        for c, l in prov_lineage.items():
            groups.setdefault(l, []).append(c)
        print("| lineage | cell types |")
        print("|--|--|")
        for l in sorted(groups):
            print(f"| {l} | {', '.join(sorted(groups[l]))} |")


if __name__ == '__main__':
    main()
