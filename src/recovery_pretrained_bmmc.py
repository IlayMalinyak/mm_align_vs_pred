"""Phase read-off (k̂, r̂, Δ̂, regime) for the PRETRAINED-ADT BMMC variant,
on current features. Z1 = scGPT RNA embeddings, Z2 = pretrained ADTEncoder
embeddings (512-d) instead of PCA on raw CLR ADT.

Uses the canonical direct/elbow estimator (recovery_analysis.estimate_cell,
ε=1e-6, PCA-100 prereduce + center, n=5000 subsample) — identical to the
raw-ADT reconciliation, so numbers are directly comparable.

Report: seed-42 canonical single run + 5-seed (0–4) summary.
"""
from __future__ import annotations
import sys
import numpy as np
from pathlib import Path

MM_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MM_ROOT / 'src'))
sys.path.insert(0, str(MM_ROOT / 'src/bmmc'))

import recovery_analysis as RA
from recovery_analysis import _prereduce_and_center, estimate_cell, BMMC_NCOMP

EPS = 1e-6
N = RA.BMMC_MAIN_N          # 5000
ADT_PRETRAINED = str(MM_ROOT / 'src/logs/bmmc/adt_pretrained/adt_encoder.pt')
SEEDS = [42, 0, 1, 2, 3, 4]


def load_pretrained():
    """Z1 = RNA (current scGPT), Z2 = pretrained ADT encoder embeddings, labels."""
    import importlib.util
    apd_path = MM_ROOT / 'src/bmmc/analyze_phase_diagram_bmmc.py'
    spec = importlib.util.spec_from_file_location('_apd_bmmc_local', str(apd_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Z1_full, Z2_full, labels, label_names, z2_label = mod.load_data(adt_pretrained=ADT_PRETRAINED)
    return (np.asarray(Z1_full, float), np.asarray(Z2_full, float),
            np.asarray(labels, float), z2_label)


def run():
    Z1_full, Z2_full, labels, z2_label = load_pretrained()
    Ntot = Z1_full.shape[0]
    print(f"# pretrained-ADT phase read-off  (Z2 = {z2_label}, dim={Z2_full.shape[1]})")
    print(f"# ε={EPS}, PCA-100 prereduce+center, n_comp={BMMC_NCOMP}, subsample n={N}/{Ntot}\n")

    rows = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(Ntot, N, replace=False))
        Z1s, Z2s, lab = Z1_full[idx], Z2_full[idx], labels[idx]
        Z1c, Z2c = _prereduce_and_center(Z1s, Z2s)
        r = estimate_cell(Z1c, Z2c, lab, EPS, BMMC_NCOMP)
        kca = int(r['cca_is_signal'].sum()); kcp = int(r['asvd_is_signal'].sum())
        rows[seed] = dict(
            kca=kca, rca=int(r['cca_rec']['r_hat']), dca=float(r['delta_ca']), bpca=int(r['cca_breakpoint']),
            kcp=kcp, rcp=int(r['asvd_rec']['r_hat']), dcp=float(r['delta_cp']), bpcp=int(r['asvd_breakpoint']),
            regime=r['regime'])
        s = rows[seed]
        print(f"[seed {seed:>2}] k̂_CA={s['kca']:>2} r̂_CA={s['rca']:>2} Δ̂_CA={s['dca']:.3f} bp_CA={s['bpca']:>2} | "
              f"k̂_CP={s['kcp']:>2} r̂_CP={s['rcp']:>2} Δ̂_CP={s['dcp']:.3f} bp_CP={s['bpcp']:>2} | {s['regime']}")

    print("\n## seed-42 canonical")
    s = rows[42]
    print(f"| k̂_CA | r̂_CA | Δ̂_CA | bp_CA | k̂_CP | r̂_CP | Δ̂_CP | bp_CP | regime |")
    print(f"|--:|--:|--:|--:|--:|--:|--:|--:|--|")
    print(f"| {s['kca']} | {s['rca']} | {s['dca']:.3f} | {s['bpca']} | {s['kcp']} | {s['rcp']} | {s['dcp']:.3f} | {s['bpcp']} | {s['regime']} |")

    ss = [rows[k] for k in [0, 1, 2, 3, 4]]
    def ms(key):
        v = np.array([x[key] for x in ss], float); return f"{v.mean():.3f} ± {v.std():.3f}"
    def msi(key):
        v = np.array([x[key] for x in ss], float); return f"{v.mean():.1f} ± {v.std():.1f}"
    print("\n## 5-seed summary (seeds 0–4)")
    print(f"| quantity | CA | CP |")
    print(f"|--|--|--|")
    print(f"| k̂ | {msi('kca')} | {msi('kcp')} |")
    print(f"| r̂ | {msi('rca')} | {msi('rcp')} |")
    print(f"| Δ̂ | {ms('dca')} | {ms('dcp')} |")
    print(f"| elbow breakpoint | {msi('bpca')} | {msi('bpcp')} |")
    regs = [x['regime'] for x in ss]
    print(f"| regime | {regs.count('Neither')}/5 Neither | — |")
    print("\nper-seed Δ̂_CA/Δ̂_CP:", ", ".join(
        f"s{k} {rows[k]['dca']:.3f}/{rows[k]['dcp']:.3f}" for k in [0,1,2,3,4]))
    print("per-seed r̂_CA/r̂_CP:", ", ".join(
        f"s{k} {rows[k]['rca']}/{rows[k]['rcp']}" for k in [0,1,2,3,4]))


if __name__ == '__main__':
    run()
