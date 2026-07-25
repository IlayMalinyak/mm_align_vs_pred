"""
Plot phase estimation sweep: Δ_CA and Δ_CP vs n_cells (5 seeds each).
Reads from src/logs/bmmc/phase_sweep/ncells_*/seed_*/bmmc_result.json
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

MM_ROOT  = Path(__file__).resolve().parent.parent.parent
SWEEP_DIR = MM_ROOT / 'src/logs/bmmc/phase_sweep'
OUT_PATH  = MM_ROOT / 'src/logs/bmmc/phase_sweep_plot.png'

SIZES = [1000, 2000, 5000]
N_SEEDS = 5

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'mathtext.fontset': 'cm',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 1.4,
})

COLOR_CA = '#2196F3'
COLOR_CP = '#F44336'

fig, ax = plt.subplots(figsize=(7, 5))

for color, delta_key, label in [
    (COLOR_CA, 'delta_ca_direct', r'$\Delta_{\mathrm{CA}}$'),
    (COLOR_CP, 'delta_cp_direct', r'$\Delta_{\mathrm{CP}}$'),
]:
    means, stds, xs = [], [], []
    for size in SIZES:
        vals = []
        for seed in range(N_SEEDS):
            p = SWEEP_DIR / f'ncells_{size}' / f'seed_{seed}' / 'bmmc_result.json'
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                vals.append(d[delta_key])
        if vals:
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            xs.append(size)
            print(f"{label} n={size}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")

    xs = np.array(xs)
    means = np.array(means)
    stds = np.array(stds)
    ax.plot(xs, means, 'o-', color=color, linewidth=2.5, markersize=9,
            markeredgewidth=2, markeredgecolor='white', label=label, zorder=3)
    ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.18, zorder=2)

ax.axhline(1.0, color='black', linewidth=1.5, linestyle='--', label=r'$\Delta = 1$ (threshold)')
ax.set_xlabel('Number of cells', fontsize=18, fontweight='bold')
ax.set_ylabel(r'$\Delta$ (direct)', fontsize=18, fontweight='bold')
ax.set_title('Phase estimation stability\nBMMC RNA (scGPT) × ADT', fontsize=18, fontweight='bold')
ax.set_xticks(SIZES)
ax.set_xticklabels([str(s) for s in SIZES], fontsize=14)
ax.tick_params(labelsize=14)
ax.legend(fontsize=14, frameon=False)
ax.spines['left'].set_linewidth(1.4)
ax.spines['bottom'].set_linewidth(1.4)

fig.tight_layout()
fig.savefig(str(OUT_PATH), dpi=200, bbox_inches='tight')
print(f'\nSaved: {OUT_PATH}')
