"""
Astro reference for the BMMC floor-attribution study (Experiment 1, step 4).

Reproduces the LAMOST x Kepler phase estimate (stored: Delta_CA=1.128,
Delta_CP=2.220, regime "Both") by re-deriving the 891-star pairing, then runs
the SAME Algorithm-1 estimator used for BMMC and attributes the nuisance-floor
component against stellar covariates (Teff, [Fe/H], Prot, magnitude, SNR).

Read-only w.r.t. the astro estimation path: it loads the cached embeddings and
prediction CSVs and the bridge parquet, but does not modify anything under
MultiDESA. Kepler is the well-behaved reference case, so the floor should map
cleanly to a stellar covariate (or show it does not, plainly).

Usage:
    python -m src.astro.floor_attribution_astro
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

MM_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MM_ROOT / 'src'))
sys.path.insert(0, str(MM_ROOT / 'src/bmmc'))

from analyze_phase_diagram import estimate_parameters, _make_cv_folds
from floor_attribution import (cca_scores, asvd_scores, attribute, role_map,
                               _zscore, BAND_K)
from sklearn.decomposition import PCA

# ── paths (read-only) ────────────────────────────────────────────────────────
MD = Path('/home/ilay.kamai/work/MultiDESA')
LAMOST_DIR = MD / 'logs/lamost/2026-02-12/44644'
# Standardized estimation set: the restored 934-star Kepler extraction
# (the original 2026-02-12-07-02 dir was overwritten post-artifact → drifted
# 879-pair set). Override with KEPLER_LOGDIR to point elsewhere.
KEPLER_DIR = Path(os.environ.get(
    'KEPLER_LOGDIR', str(MD / 'logs/kepler/2026-02-12-07-02_restore891')))
LAMOST_CSV = LAMOST_DIR / 'lamost_SpectralViT_preds.csv'
KEPLER_CSV = KEPLER_DIR / 'kepler_dualformer_dualvar_no_cov_DoubleInputRegressor_preds.csv'
BRIDGE     = Path('/home/ilay.kamai/work/multimodal/multimodal_df_min2.parquet')
OUT_DIR    = MM_ROOT / 'src/logs/bmmc/floor_attribution'
OUT_DIR.mkdir(parents=True, exist_ok=True)
# Suffix for output filenames (e.g. '_934') so reruns don't overwrite the
# stored reference outputs.
OUT_SUFFIX = os.environ.get('ASTRO_FLOOR_SUFFIX', '')

PCA_PREREDUCE = 100
N_COMPONENTS  = 50
N_PERMUTATIONS = 30


def _ids(s):
    if s.dtype.kind == 'f':
        return s.astype('Int64')
    return s


def load_pairs():
    U = np.load(LAMOST_DIR / 'svd_features/U_test.npy')
    S = np.load(LAMOST_DIR / 'svd_features/S_test.npy')
    Z_la = (U * S[np.newaxis, :]).astype(np.float32)
    df_la = pd.read_csv(LAMOST_CSV)
    Z_ke = np.load(KEPLER_DIR / 'svd_features/features_output_space.npy').astype(np.float32)
    df_ke = pd.read_csv(KEPLER_CSV)
    print(f"LAMOST Z {Z_la.shape} / csv {len(df_la)};  Kepler Z {Z_ke.shape} / csv {len(df_ke)}")
    # trim CSV to embedding length (as load_data does)
    df_la = df_la.iloc[:len(Z_la)].reset_index(drop=True)
    df_ke = df_ke.iloc[:len(Z_ke)].reset_index(drop=True)

    mdf = pd.read_parquet(BRIDGE, columns=['lamost_obsid', 'KID'])
    # first-occurrence index maps
    a_map = {}
    for i, v in _ids(df_la['obsid']).items():
        if pd.notna(v) and v not in a_map:
            a_map[int(v)] = i
    b_map = {}
    for i, v in _ids(df_ke['KID']).items():
        if pd.notna(v) and v not in b_map:
            b_map[int(v)] = i

    mask = mdf['lamost_obsid'].notna() & mdf['KID'].notna()
    mdf = mdf[mask]
    idx_a, idx_b = [], []
    for va, vb in zip(_ids(mdf['lamost_obsid']).values, _ids(mdf['KID']).values):
        if pd.isna(va) or pd.isna(vb):
            continue
        va, vb = int(va), int(vb)
        if va in a_map and vb in b_map:
            idx_a.append(a_map[va]); idx_b.append(b_map[vb])
    idx_a = np.array(idx_a); idx_b = np.array(idx_b)
    print(f"  matched pairs: {len(idx_a)}")

    Z1 = Z_la[idx_a]
    Z2 = Z_ke[idx_b]
    la = df_la.iloc[idx_a].reset_index(drop=True)
    ke = df_ke.iloc[idx_b].reset_index(drop=True)

    labels = la['logg'].to_numpy(np.float64).reshape(-1, 1)  # shared label = logg

    # stellar covariates (row-aligned)
    cov = {
        'logg_sharedlabel': la['logg'],           # the label itself (reference)
        'Teff':  la['teff'],
        'FeH':   la['feh'],
        'SNR':   la['snr'],
        'Prot':  ke['Prot'],
        'Kepmag': ke['KEPMAG'],
        'M_K':   ke['kmag_abs'],
    }
    return Z1, Z2, labels, cov


def _cov_design(cov):
    """z-scored 1-column design per covariate; NaN -> 0 (mean). Report n_valid."""
    out = []
    for name, s in cov.items():
        x = np.asarray(s, dtype=np.float64)
        valid = np.isfinite(x)
        xf = x.copy()
        if valid.any():
            xf[~valid] = np.nanmean(x[valid])
        std = xf.std()
        z = (xf - xf.mean()) / std if std > 1e-12 else xf - xf.mean()
        out.append((name, 'stellar', z.reshape(-1, 1), int(valid.sum())))
    return out


def main():
    Z1, Z2, labels, cov = load_pairs()

    k1 = min(PCA_PREREDUCE, Z1.shape[1], Z1.shape[0] - 1)
    k2 = min(PCA_PREREDUCE, Z2.shape[1], Z2.shape[0] - 1)
    Z1c = PCA(n_components=k1, random_state=0).fit_transform(Z1); Z1c -= Z1c.mean(0)
    Z2c = PCA(n_components=k2, random_state=0).fit_transform(Z2); Z2c -= Z2c.mean(0)

    nc = min(N_COMPONENTS, Z1c.shape[1], Z2c.shape[1])
    print(f"\nEstimating (Algorithm 1): Z1c{Z1c.shape} Z2c{Z2c.shape} n_comp={nc}")
    res = estimate_parameters(Z1c, Z2c, labels, n_components=nc, cv=5,
                              n_permutations=N_PERMUTATIONS, classification_method='elbow')
    dca, dcp = res['delta_ca_direct'], res['delta_cp_direct']
    cca, cpd = res['cca'], res['cp_direct']
    print(f"\n  REPRODUCED astro: Delta_CA={dca:.4f} Delta_CP={dcp:.4f} regime={res['regime_primary']}"
          f"  (stored: 1.128 / 2.220 / Both)")
    print(f"  n_signal CCA={cca['n_signal']} A-SVD={cpd['n_signal']}")

    phi, cca_x, cca_y = cca_scores(Z1c, Z2c, nc=nc)
    sigma, asvd_x = asvd_scores(Z1c, Z2c, nc=nc)
    assert np.allclose(phi, cca['canonical_correlations'], atol=1e-6)
    assert np.allclose(sigma, cpd['singular_values'], atol=1e-6)

    covariates = _cov_design(cov)
    cov_named = [(n, k, X) for (n, k, X, _) in covariates]
    nvalid = {n: v for (n, _, _, v) in covariates}
    folds = _make_cv_folds(Z1c.shape[0], 5)
    ct_r2_cca = cca['r2_matrix'].sum(axis=1)
    ct_r2_asvd = cpd['r2_matrix'].sum(axis=1)

    rows = []
    for decomp, spec, is_sig, sx, sy, ctr2 in [
        ('CCA', phi, cca['is_signal'], cca_x, cca_y, ct_r2_cca),
        ('A-SVD', sigma, cpd['is_signal'], asvd_x, None, ct_r2_asvd),
    ]:
        roles = role_map(spec, is_sig)
        band = list(np.argsort(spec)[::-1][:min(BAND_K, len(spec))])
        for j in sorted(set(int(x) for x in band) | set(roles.keys())):
            sides = {'lamost': sx[:, j]}
            if sy is not None:
                sides['kepler'] = sy[:, j]
            for side, sc in sides.items():
                r2 = attribute(sc, cov_named, folds)
                row = {'decomposition': decomp, 'component': j,
                       'is_signal': bool(is_sig[j]), 'role': roles.get(j, ''),
                       'score_side': side, 'singular_value': float(spec[j]),
                       'sumR2_vs_logg': float(ctr2[j])}
                row.update({f'R2__{k}': r2[k] for k in r2})
                rows.append(row)
    df = pd.DataFrame(rows).sort_values(['decomposition', 'component', 'score_side'])
    csv = OUT_DIR / f'astro_floor_attribution{OUT_SUFFIX}.csv'
    df.to_csv(csv, index=False)
    print(f"\nSaved {csv}   (n_valid per covariate: {nvalid})")

    # floor identity
    fc = int(np.where(~cca['is_signal'])[0][np.argmax(phi[~cca['is_signal']])]) if (~cca['is_signal']).any() else -1
    fa = int(np.where(~cpd['is_signal'])[0][np.argmax(sigma[~cpd['is_signal']])]) if (~cpd['is_signal']).any() else -1
    print(f"  floor: CCA=comp{fc} (phi={phi[fc]:.3f})  A-SVD=comp{fa} (sigma={sigma[fa]:.3f})")

    cols = ['Teff', 'FeH', 'SNR', 'Prot', 'Kepmag', 'M_K', 'logg_sharedlabel']
    print("\n  decomp cmp sig role              side    sv  logg | " + " ".join(f"{c[:7]:>7}" for c in cols))
    for _, r in df.iterrows():
        print(f"  {r['decomposition']:5} {int(r['component']):>3} {int(bool(r['is_signal']))} "
              f"{r['role'][:16]:16} {r['score_side']:6} {r['singular_value']:5.2f} {r['sumR2_vs_logg']:4.2f} | "
              + " ".join(f"{r[f'R2__{c}']:7.3f}" for c in cols))

    with open(OUT_DIR / f'astro_meta{OUT_SUFFIX}.json', 'w') as fh:
        json.dump({'delta_ca_direct': dca, 'delta_cp_direct': dcp,
                   'regime': res['regime_primary'],
                   'n_signal_cca': int(cca['n_signal']), 'n_signal_asvd': int(cpd['n_signal']),
                   'floor_cca': fc, 'floor_sv_cca': float(phi[fc]) if fc >= 0 else None,
                   'floor_asvd': fa, 'floor_sv_asvd': float(sigma[fa]) if fa >= 0 else None,
                   'n_pairs': int(Z1.shape[0]), 'n_valid': nvalid}, fh, indent=2)


if __name__ == '__main__':
    main()
