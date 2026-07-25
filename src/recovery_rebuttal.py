"""
Rebuttal analysis for the ε×n sensitivity sweep.

TASK A — resampling-scheme diagnostic (LAMOST×Kepler, ε=1e-6):
  arm1 bootstrap WITH replacement n=N, arm2 subsample WITHOUT replacement m=591
  (= arm1 expected distinct count), arm3 subsample WITHOUT replacement m=750.
TASK B — full sweep rebuilt with SUBSAMPLING WITHOUT REPLACEMENT, 50 replicates,
  md5 seeds, per-cell regime fractions + m=N deterministic reference rows.

Emits markdown tables to stdout. No plots.

    python -m src.recovery_rebuttal --task A
    python -m src.recovery_rebuttal --task B --datasets kepler tess bmmc
    python -m src.recovery_rebuttal --task AB
"""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter
import numpy as np

from src import recovery_analysis as RA
from src.recovery_analysis import (
    load_astro, load_bmmc, _prereduce_and_center, estimate_cell, _regime)

EPS_GRID = [1e-6, 1e-4, 1e-2, 1e-1]
NREP = 50
REGIMES = ['Both', 'CA only', 'CP only', 'Neither']

M_LADDER = {
    'kepler': [250, 500, 750],
    'tess':   [80, 110, 140],
    'bmmc':   [250, 500, 1000, 5000, 72000],
}


def _seed(*parts):
    return int(hashlib.md5('|'.join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def _mi(vals, dec=2):
    a = np.asarray([v for v in vals if v is not None], float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return 'nan'
    m, lo, hi = np.median(a), np.percentile(a, 25), np.percentile(a, 75)
    f = f'{{:.{dec}f}}'
    return f'{f.format(m)} [{f.format(lo)},{f.format(hi)}]'


def _regime_fracs(cells):
    c = Counter(r['regime'] for r in cells)
    n = len(cells)
    return ' / '.join(f'{k} {c[k]/n:.2f}' for k in REGIMES if c[k] > 0)


def _one(ds, idx, reg):
    Z1c, Z2c = _prereduce_and_center(ds['X'][idx], ds['Y'][idx])
    return estimate_cell(Z1c, Z2c, ds['labels'][idx], reg, ds['n_components'])


# ── TASK A ──────────────────────────────────────────────────────────────────
def task_A():
    ds = load_astro('kepler')
    N = ds['X'].shape[0]
    arms = [('arm1 bootstrap w/repl n=N', N, True),
            ('arm2 subsample wo/repl m=591', 591, False),
            ('arm3 subsample wo/repl m=750', 750, False)]
    print(f"\n## TASK A — resampling-scheme diagnostic (LAMOST × Kepler, ε=1e-6, N={N}, "
          f"{NREP} replicates/arm)\n")
    summary_rows, r2_by_arm, bp_by_arm, sigfrac_by_arm = [], {}, {}, {}
    for label, m, replace in arms:
        cells, r2s, bps, sigs = [], [], [], []
        for b in range(NREP):
            rng = np.random.default_rng(_seed('A', label, m, replace, b))
            idx = rng.choice(N, m, replace=replace)
            r = _one(ds, idx, 1e-6)
            cells.append(r)
            r2s.append(r['cca_sum_r2'][:16])
            bps.append(r['cca_breakpoint'])
            sigs.append(r['cca_is_signal'][:16].astype(float))
        summary_rows.append((label, m, replace, cells))
        r2_by_arm[label] = np.median(np.array(r2s), axis=0)
        bp_by_arm[label] = int(np.median(bps))
        sigfrac_by_arm[label] = np.mean(np.array(sigs), axis=0)

    print("### A.1 — median [IQR] over 50 replicates\n")
    print("| arm | m | replace | distinct(med) | Δ̂_CA | Δ̂_CP | r̂_CA | r̂_CP | k̂_CA | k̂_CP |")
    print("|--|--:|--|--:|--|--|--|--|--|--|")
    for label, m, replace, cells in summary_rows:
        distinct = int(np.median([len(np.unique(np.random.default_rng(_seed('A', label, m, replace, b)).choice(N, m, replace=replace))) for b in range(NREP)]))
        print(f"| {label} | {m} | {replace} | {distinct} | "
              f"{_mi([c['delta_ca'] for c in cells])} | {_mi([c['delta_cp'] for c in cells])} | "
              f"{_mi([c['cca_rec']['r_hat'] for c in cells],0)} | {_mi([c['asvd_rec']['r_hat'] for c in cells],0)} | "
              f"{_mi([c['cca_rec']['k_hat'] for c in cells],0)} | {_mi([c['asvd_rec']['k_hat'] for c in cells],0)} |")

    print("\n### A.2 — median per-component CCA CV R² (components 0–15) + median elbow breakpoint\n")
    labels = [l for l, *_ in summary_rows]
    print("| CCA comp | " + " | ".join(l.split()[0] for l in labels) + " |")
    print("|--:|" + "|".join(["--"] * len(labels)) + "|")
    for i in range(16):
        print(f"| {i} | " + " | ".join(f"{r2_by_arm[l][i]:+.5f}" for l in labels) + " |")
    print(f"| **elbow breakpoint (med)** | " + " | ".join(str(bp_by_arm[l]) for l in labels) + " |")

    print("\n### A.3 — fraction of 50 replicates in which CCA component i is classified signal\n")
    print("| CCA comp | " + " | ".join(l.split()[0] for l in labels) + " |")
    print("|--:|" + "|".join(["--"] * len(labels)) + "|")
    for i in range(16):
        print(f"| {i} | " + " | ".join(f"{sigfrac_by_arm[l][i]:.2f}" for l in labels) + " |")


# ── TASK B ──────────────────────────────────────────────────────────────────
def _sweep_dataset(ds, key):
    N = ds['X'].shape[0]
    ladder = M_LADDER[key]
    name = ds['name']
    print(f"\n### {name}  (subsample WITHOUT replacement, N={N}, {NREP} replicates/cell)\n")
    print("| m | ε | r̂_CA | r̂_CP | k̂_CA | k̂_CP | Δ̂_CA | Δ̂_CP | regime fractions |")
    print("|--:|--:|--|--|--|--|--|--|--|")
    for m in ladder:
        # resample once per replicate, evaluate all ε on it
        boot = []
        for b in range(NREP):
            rng = np.random.default_rng(_seed('B', name, m, b))
            idx = rng.choice(N, m, replace=False)
            Z1c, Z2c = _prereduce_and_center(ds['X'][idx], ds['Y'][idx])
            boot.append((Z1c, Z2c, ds['labels'][idx]))
        for reg in EPS_GRID:
            cells = [estimate_cell(Z1c, Z2c, lab, reg, ds['n_components']) for (Z1c, Z2c, lab) in boot]
            print(f"| {m} | {reg:.0e} | "
                  f"{_mi([c['cca_rec']['r_hat'] for c in cells],0)} | {_mi([c['asvd_rec']['r_hat'] for c in cells],0)} | "
                  f"{_mi([c['cca_rec']['k_hat'] for c in cells],0)} | {_mi([c['asvd_rec']['k_hat'] for c in cells],0)} | "
                  f"{_mi([c['delta_ca'] for c in cells])} | {_mi([c['delta_cp'] for c in cells])} | "
                  f"{_regime_fracs(cells)} |")
    # m=N deterministic reference (no resample)
    Z1c, Z2c = _prereduce_and_center(ds['X'], ds['Y'])
    print(f"\n_Reference — full sample m=N={N}, deterministic (no resample):_\n")
    print("| m | ε | r̂_CA | r̂_CP | k̂_CA | k̂_CP | Δ̂_CA | Δ̂_CP | regime |")
    print("|--:|--:|--:|--:|--:|--:|--:|--:|--|")
    for reg in EPS_GRID:
        r = estimate_cell(Z1c, Z2c, ds['labels'], reg, ds['n_components'])
        dca = 'inf' if r['delta_ca'] == float('inf') else f"{r['delta_ca']:.3f}"
        dcp = 'inf' if r['delta_cp'] == float('inf') else f"{r['delta_cp']:.3f}"
        print(f"| N={N} | {reg:.0e} | {r['cca_rec']['r_hat']} | {r['asvd_rec']['r_hat']} | "
              f"{r['cca_rec']['k_hat']} | {r['asvd_rec']['k_hat']} | {dca} | {dcp} | {r['regime']} |")


def task_B(keys):
    print("\n## TASK B — sweep with subsampling WITHOUT replacement "
          f"(ε∈{{1e-6,1e-4,1e-2,1e-1}}, {NREP} replicates, md5 seeds)")
    for key in keys:
        ds = load_astro(key) if key in ('kepler', 'tess') else load_bmmc()
        _sweep_dataset(ds, key)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['A', 'B', 'AB'], default='AB')
    p.add_argument('--datasets', nargs='*', default=['kepler', 'tess', 'bmmc'],
                   choices=['kepler', 'tess', 'bmmc'])
    args = p.parse_args()
    if args.task in ('A', 'AB'):
        task_A()
    if args.task in ('B', 'AB'):
        task_B(args.datasets)


if __name__ == '__main__':
    main()
