"""
Generate the 2x2 phase diagnostics figure for the NeurIPS appendix.

Rows: LAMOST x Kepler, LAMOST x TESS
Columns: C-hat (CCA) spectrum, A-hat (regression/CP) spectrum

Uses pre-computed classification data from analyze_phase_diagram.py
(the NPZ files with cp_direct_* and cca_* arrays).

Usage:
    python analysis/plot_phase_diagnostics.py
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent / 'multimodal_comparison'
OUT_DIR = Path(__file__).resolve().parent.parent / 'figures'

PAIRS = [
    {
        'label': r'LAMOST $\times$ Kepler',
        'short': 'Kepler',
        'npz': BASE / 'phase_lamost_x_kepler_classification_data.npz',
        'json': BASE / 'phase_lamost_x_kepler_result.json',
    },
    {
        'label': r'LAMOST $\times$ TESS',
        'short': 'TESS',
        'npz': BASE / 'phase_lamost_x_tess_classification_data.npz',
        'json': BASE / 'phase_lamost_x_tess_result.json',
    },
]

# ── Colors ───────────────────────────────────────────────────────────
C_SIGNAL = '#d4edda'       # light green
C_NUISANCE = '#f8d7da'     # light pink
C_BAR = '#999999'          # gray bars
C_R2 = '#2ca089'           # teal for R² line
C_FLOOR = '#333333'        # nuisance floor dashed line


def _compute_delta(sv, is_signal):
    """min(signal sv) / max(nuisance sv), handling edge cases."""
    is_nuisance = ~is_signal
    if not is_signal.any():
        return 0.0
    if not is_nuisance.any():
        return float('inf')
    min_sig = float(sv[is_signal].min())
    max_nui = float(sv[is_nuisance].max())
    if max_nui <= 0:
        return float('inf')
    return min_sig / max_nui


def _plot_panel(ax, sv, sum_r2, is_signal, delta_val, delta_name,
                title, show_legend=False):
    """
    One panel: gray bars (singular values, left axis) + teal R² line (right axis)
    with signal/nuisance background shading and nuisance floor line.
    """
    nc = min(50, len(sv))
    sv = sv[:nc]
    sum_r2 = sum_r2[:nc]
    is_signal = is_signal[:nc]
    is_nuisance = ~is_signal
    x = np.arange(1, nc + 1)

    # ── Background shading ───────────────────────────────────────────
    for i in range(nc):
        color = C_SIGNAL if is_signal[i] else C_NUISANCE
        ax.axvspan(i + 0.5, i + 1.5, alpha=0.4, color=color, zorder=0)

    # ── Gray bars: singular values (left y-axis) ─────────────────────
    ax.bar(x, sv, color=C_BAR, alpha=0.6, width=0.7, zorder=2,
           label=r'$\sigma_i$')
    ax.set_ylabel(r'Singular value $\sigma_i$', fontsize=10)
    ax.set_xlabel('Component index', fontsize=10)
    ax.set_xlim(0.5, nc + 0.5)

    # ── Nuisance floor line ──────────────────────────────────────────
    if is_nuisance.any():
        floor = float(sv[is_nuisance].max())
        floor_label = 'nuisance floor' if show_legend else None
        ax.axhline(floor, color=C_FLOOR, ls='--', lw=1.2, alpha=0.7,
                   zorder=4, label=floor_label)

    # ── R² line on right axis ────────────────────────────────────────
    ax2 = ax.twinx()
    ax2.plot(x, sum_r2, '-o', color=C_R2, lw=1.5, markersize=3,
             alpha=0.85, zorder=5, label=r'$R^2$ (log $g$)')
    ax2.axhline(0, color=C_R2, ls='-', lw=0.5, alpha=0.3, zorder=1)

    # Symmetric R² limits around zero if values are near zero
    r2_max = max(abs(sum_r2.max()), abs(sum_r2.min()), 0.05)
    r2_upper = max(r2_max * 1.3, 0.05)
    r2_lower = min(-0.05, sum_r2.min() * 1.3)
    ax2.set_ylim(r2_lower, r2_upper)
    ax2.set_ylabel(r'$R^2$ (per component)', fontsize=10, color=C_R2)
    ax2.tick_params(axis='y', colors=C_R2)

    # ── Title ────────────────────────────────────────────────────────
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)

    # ── Annotation box ───────────────────────────────────────────────
    n_sig = int(is_signal.sum())
    if delta_val == 0.0 and n_sig == 0:
        delta_str = r'$\hat{\Delta}_{' + delta_name + r'}$ undefined'
    elif delta_val < 1:
        delta_str = (r'$\hat{\Delta}_{' + delta_name +
                     r'}$ = ' + f'{delta_val:.2f} < 1')
    else:
        delta_str = (r'$\hat{\Delta}_{' + delta_name +
                     r'}$ = ' + f'{delta_val:.2f}')
    ann = f'#signal = {n_sig},  {delta_str}'
    ax.text(0.97, 0.95, ann, transform=ax.transAxes, fontsize=9,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#999', alpha=0.9))


def main():
    # ── Load data ────────────────────────────────────────────────────
    pair_data = []
    for pair in PAIRS:
        d = np.load(pair['npz'], allow_pickle=True)
        with open(pair['json']) as f:
            res = json.load(f)
        pair_data.append({
            'label': pair['label'],
            'short': pair['short'],
            # CCA (C-hat)
            'cca_sv': d['cca_sv'],
            'cca_r2': d['cca_r2'].sum(axis=1),  # sum across labels
            'cca_is_signal': d['cca_is_signal'],
            'cca_n_signal': int(d['cca_n_signal']),
            # CP-direct (A-hat)
            'cp_sv': d['cp_direct_sv'],
            'cp_r2': d['cp_direct_sum_r2'],
            'cp_is_signal': d['cp_direct_is_signal'],
            'cp_n_signal': int(d['cp_direct_n_signal']),
            # Deltas
            'delta_ca': res['delta_ca_direct'],
            'delta_cp': res['delta_cp_direct'],
        })

    # ── Verify deltas match recomputation from spectra ───────────────
    for pd_ in pair_data:
        recomp_ca = _compute_delta(pd_['cca_sv'], pd_['cca_is_signal'])
        recomp_cp = _compute_delta(pd_['cp_sv'], pd_['cp_is_signal'])
        print(f"{pd_['short']}: CCA #signal={pd_['cca_n_signal']}, "
              f"CP #signal={pd_['cp_n_signal']}")
        print(f"  delta_CA: JSON={pd_['delta_ca']:.4f}, "
              f"recomputed={recomp_ca:.4f}")
        print(f"  delta_CP: JSON={pd_['delta_cp']:.4f}, "
              f"recomputed={recomp_cp:.4f}")

    # ── Figure setup ─────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 10,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for row_idx, pd_ in enumerate(pair_data):
        # Column 0: C-hat (CCA)
        _plot_panel(
            axes[row_idx, 0],
            sv=pd_['cca_sv'],
            sum_r2=pd_['cca_r2'],
            is_signal=pd_['cca_is_signal'],
            delta_val=pd_['delta_ca'],
            delta_name='CA',
            title=f"{pd_['label']}: " + r'$\hat{C}$ (CCA)',
            show_legend=(row_idx == 0),
        )
        # Column 1: A-hat (CP regression)
        _plot_panel(
            axes[row_idx, 1],
            sv=pd_['cp_sv'],
            sum_r2=pd_['cp_r2'],
            is_signal=pd_['cp_is_signal'],
            delta_val=pd_['delta_cp'],
            delta_name='CP',
            title=f"{pd_['label']}: " + r'$\hat{A}$ (CP)',
            show_legend=False,
        )

    # ── Shared legend at bottom ──────────────────────────────────────
    legend_elements = [
        Patch(facecolor=C_SIGNAL, edgecolor='#999', alpha=0.6,
              label='Signal'),
        Patch(facecolor=C_NUISANCE, edgecolor='#999', alpha=0.6,
              label='Nuisance'),
        Patch(facecolor=C_BAR, edgecolor='#999', alpha=0.6,
              label=r'$\sigma_i$'),
        plt.Line2D([0], [0], color=C_R2, lw=1.5, marker='o',
                   markersize=3, label=r'Per-component $R^2$'),
        plt.Line2D([0], [0], color=C_FLOOR, ls='--', lw=1.2,
                   label='Nuisance floor'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=10, frameon=True, edgecolor='#ccc',
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Signal vs nuisance spectra underlying the regime predictions',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])

    # ── Save ─────────────────────────────────────────────────────────
    out_png = OUT_DIR / 'phase_diagnostics.png'
    out_pdf = OUT_DIR / 'phase_diagnostics.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")

    # ── Summary confirmation ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONFIRMATION — values for Table 1")
    print("=" * 60)
    for pd_ in pair_data:
        print(f"\n{pd_['short']}:")
        print(f"  C-hat (CCA):  #signal = {pd_['cca_n_signal']}, "
              f"delta_CA = {pd_['delta_ca']:.4f}")
        print(f"  A-hat (CP):   #signal = {pd_['cp_n_signal']}, "
              f"delta_CP = {pd_['delta_cp']:.4f}")


if __name__ == '__main__':
    main()
