"""
Parse SLURM array logs from bmmc_cross_modal, compute mean ± std across seeds,
generate bar plot and print LaTeX table.

Usage:
    python -m src.bmmc.plot_results \
        --log_pattern '/home/ilay.kamai/athena/logs/bmmc_cross_modal_115682_*.out'
"""
from __future__ import annotations
import argparse
import glob
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Output ────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent.parent / 'logs/bmmc'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Representations and display names
REPS = ['rna_only', 'adt_only', 'frozen_concat', 'ca', 'cp', 'cp_reverse']
REP_LABELS = {
    'rna_only':    'RNA only',
    'adt_only':    'ADT only',
    'frozen_concat': 'Frozen concat',
    'ca':          'CA (trained)',
    'cp':          'CP (trained)',
    'cp_reverse':  'CP\u2190 (trained)',
}

MODES = ['CA', 'CP', 'CP_rev']
MODE_COLORS = {'CA': '#4A90C4', 'CP': '#E07B54', 'CP_rev': '#7B6D8E'}


def parse_log(path: str) -> dict | None:
    """Parse balanced_accuracy summary from one log file."""
    results = {}
    in_summary = False
    with open(path) as f:
        for line in f:
            if 'SUMMARY' in line and 'balanced_accuracy' in line:
                in_summary = True
                continue
            if in_summary:
                if '===' in line:
                    if results:
                        break
                    continue
                # match lines like:  rna_only   0.527    0.527    0.527
                m = re.match(
                    r'\s+(\S+)\s+([\d.]+|nan)\s+([\d.]+|nan)\s+([\d.]+|nan)', line)
                if m:
                    rep = m.group(1)
                    vals = [float(v) if v != 'nan' else np.nan
                            for v in [m.group(2), m.group(3), m.group(4)]]
                    results[rep] = vals  # [CA, CP, CP_rev]
    return results if results else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--log_pattern', type=str,
                   default='/home/ilay.kamai/athena/logs/bmmc_cross_modal_115682_*.out')
    args = p.parse_args()

    files = sorted(glob.glob(args.log_pattern))
    if not files:
        raise FileNotFoundError(f"No files matching: {args.log_pattern}")
    print(f"Found {len(files)} log files.")

    # Collect per-seed results
    all_results = []
    for f in files:
        r = parse_log(f)
        if r is None:
            print(f"  WARNING: could not parse {f}")
        else:
            all_results.append(r)
    print(f"  Parsed {len(all_results)} valid logs.")

    # Aggregate: mean ± std per (rep, mode)
    # Shape: [n_seeds, n_reps, n_modes]
    n_seeds = len(all_results)
    data = np.full((n_seeds, len(REPS), len(MODES)), np.nan)
    for s, r in enumerate(all_results):
        for ri, rep in enumerate(REPS):
            if rep in r:
                data[s, ri, :] = r[rep]

    means = np.nanmean(data, axis=0)   # (n_reps, n_modes)
    stds  = np.nanstd(data, axis=0)    # (n_reps, n_modes)

    # ── Print table ───────────────────────────────────────────────────────
    print("\n" + "="*72)
    print(f"  BMMC Results — balanced accuracy (mean ± std, n={n_seeds} seeds)")
    print("="*72)
    header = f"  {'Representation':<22}{'CA':>16}{'CP':>16}{'CP_rev':>16}"
    print(header)
    print("  " + "-"*70)
    for ri, rep in enumerate(REPS):
        row = f"  {REP_LABELS[rep]:<22}"
        for mi in range(len(MODES)):
            m, s = means[ri, mi], stds[ri, mi]
            if np.isnan(m):
                row += f"{'—':>16}"
            else:
                row += f"  {m:.3f} ± {s:.3f}  "
        print(row)
    print("="*72)

    # ── LaTeX table ───────────────────────────────────────────────────────
    print("\nLaTeX table rows:")
    for ri, rep in enumerate(REPS):
        row = REP_LABELS[rep]
        for mi in range(len(MODES)):
            m, s = means[ri, mi], stds[ri, mi]
            if np.isnan(m):
                row += " & —"
            else:
                row += f" & ${m:.3f} \\pm {s:.3f}$"
        print(row + " \\\\")

    # ── Bar plot ──────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'axes.linewidth': 1.4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    # Focus plot: trained representations + unimodal baselines (no frozen_concat — unfair dim)
    plot_reps  = ['rna_only', 'adt_only', 'ca', 'cp', 'cp_reverse']
    plot_ri    = [REPS.index(r) for r in plot_reps]
    plot_labels = [REP_LABELS[r] for r in plot_reps]

    x      = np.arange(len(plot_reps))
    width  = 0.25
    offsets = [-width, 0, width]

    fig, ax = plt.subplots(figsize=(9, 5))

    for mi, (mode, color) in enumerate(MODE_COLORS.items()):
        m_vals = means[plot_ri, mi]
        s_vals = stds[plot_ri, mi]
        mask   = ~np.isnan(m_vals)
        bars = ax.bar(x[mask] + offsets[mi], m_vals[mask], width,
                      yerr=s_vals[mask], capsize=4,
                      color=color, alpha=0.85, label=mode,
                      error_kw=dict(elinewidth=1.4, ecolor='#333'))

    ax.set_xticks(x)
    ax.set_xticklabels(plot_labels, fontsize=14)
    ax.set_ylabel('Balanced Accuracy', fontsize=16, fontweight='bold')
    ax.set_title('BMMC CITE-seq: RNA × ADT', fontsize=18, fontweight='bold')
    ax.legend(fontsize=13, framealpha=0.9)
    ax.tick_params(labelsize=13)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.1)

    # Mark best trained method per group
    trained_mi = {'ca': 0, 'cp': 1, 'cp_reverse': 2}
    for ri_idx, rep in enumerate(plot_reps):
        if rep in trained_mi:
            mi = trained_mi[rep]
            m = means[REPS.index(rep), mi]
            s = stds[REPS.index(rep), mi]
            if not np.isnan(m):
                ax.text(x[ri_idx] + offsets[mi], m + s + 0.01,
                        f'{m:.3f}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')

    fig.tight_layout()
    out_path = OUT_DIR / 'bmmc_results.png'
    fig.savefig(str(out_path), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved plot to: {out_path}")


if __name__ == '__main__':
    main()
