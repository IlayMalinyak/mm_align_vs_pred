"""
Single-job phase estimation sweep: 3 cell counts × 5 seeds = 15 runs.
Loads data once, then runs all combinations in parallel via multiprocessing.
Generates summary plot at the end.
"""
from __future__ import annotations

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count

MM_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MM_ROOT / 'src'))
sys.path.insert(0, str(MM_ROOT / 'src/bmmc'))

from analyze_phase_diagram_bmmc import load_data, run

SIZES   = [1000, 2000, 5000]
N_SEEDS = 5
N_WORKERS = min(cpu_count(), 8)

SWEEP_DIR = MM_ROOT / 'src/logs/bmmc/phase_sweep'
OUT_PLOT  = MM_ROOT / 'src/logs/bmmc/phase_sweep_plot.png'

plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'mathtext.fontset': 'cm',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'axes.linewidth': 1.4,
})


def _worker(kwargs):
    """Top-level callable for multiprocessing (must be picklable)."""
    Z1      = kwargs.pop('Z1')
    Z2      = kwargs.pop('Z2')
    labels  = kwargs.pop('labels')
    lnames  = kwargs.pop('label_names')
    try:
        result = run(Z1, Z2, labels, lnames, **kwargs)
        return kwargs['max_cells'], kwargs['seed'], result
    except Exception as e:
        print(f"  ERROR n={kwargs['max_cells']} s={kwargs['seed']}: {e}")
        return kwargs['max_cells'], kwargs['seed'], None


def plot_sweep():
    fig, ax = plt.subplots(figsize=(7, 5))

    COLOR_CA = '#2196F3'
    COLOR_CP = '#F44336'

    for color, key, label in [
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
                        vals.append(json.load(f)[key])
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                xs.append(size)
                print(f"{label}  n={size}: {np.mean(vals):.3f} ± {np.std(vals):.3f}  (n={len(vals)})")

        xs = np.array(xs); means = np.array(means); stds = np.array(stds)
        ax.plot(xs, means, 'o-', color=color, linewidth=2.5, markersize=9,
                markeredgewidth=2, markeredgecolor='white', label=label, zorder=3)
        ax.fill_between(xs, means - stds, means + stds, color=color, alpha=0.18, zorder=2)

    ax.axhline(1.0, color='black', linewidth=1.5, linestyle='--',
               label=r'$\Delta = 1$ (threshold)')
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
    fig.savefig(str(OUT_PLOT), dpi=200, bbox_inches='tight')
    print(f'\nSaved plot: {OUT_PLOT}')


def main():
    print(f"Loading data once...")
    Z1, Z2, labels, label_names, _ = load_data()
    print(f"Data loaded: Z1={Z1.shape}, Z2={Z2.shape}\n")

    jobs = []
    for size in SIZES:
        for seed in range(N_SEEDS):
            out_dir = SWEEP_DIR / f'ncells_{size}' / f'seed_{seed}'
            jobs.append(dict(
                Z1=Z1, Z2=Z2, labels=labels, label_names=label_names,
                max_cells=size, seed=seed, out_dir=str(out_dir),
                pca_components=100, n_components=100, n_permutations=30,
            ))

    print(f"Running {len(jobs)} jobs with {N_WORKERS} workers...\n")
    with Pool(processes=N_WORKERS) as pool:
        pool.map(_worker, jobs)

    print("\n" + "="*60)
    print("All jobs done. Generating sweep plot...")
    plot_sweep()


if __name__ == '__main__':
    main()
