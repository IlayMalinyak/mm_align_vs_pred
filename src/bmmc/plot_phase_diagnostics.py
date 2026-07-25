"""
Generate the phase diagnostics figure for BMMC CITE-seq (RNA × ADT).

1×2 layout: Ĉ (CCA) spectrum | Â (CP regression) spectrum

Uses pre-computed classification data from analyze_phase_diagram_bmmc.py
(the NPZ and JSON files in logs/bmmc/phase_diagram/).

Usage:
    python analysis/plot_phase_diagnostics_bmmc.py
"""
from __future__ import annotations
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent.parent / 'src/logs/bmmc/phase_diagram'
OUT_DIR  = Path(__file__).resolve().parent.parent.parent / 'src/logs/bmmc/phase_diagram'

NPZ_PATH  = DATA_DIR / 'bmmc_classification_data.npz'
JSON_PATH = DATA_DIR / 'bmmc_result.json'

PAIR_LABEL = r'RNA (scGPT) $\times$ ADT (CLR)'

# ── Colors (same as astro) ─────────────────────────────────────────────────
C_SIGNAL   = '#d4edda'
C_NUISANCE = '#f8d7da'
C_BAR      = '#999999'
C_R2       = '#2ca089'
C_FLOOR    = '#333333'


def _compute_delta(sv, is_signal):
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
    nc = min(50, len(sv))
    sv       = sv[:nc]
    sum_r2   = sum_r2[:nc]
    is_signal = is_signal[:nc]
    is_nuisance = ~is_signal
    x = np.arange(1, nc + 1)

    # Background shading
    for i in range(nc):
        color = C_SIGNAL if is_signal[i] else C_NUISANCE
        ax.axvspan(i + 0.5, i + 1.5, alpha=0.4, color=color, zorder=0)

    # Gray bars: singular values (left axis)
    ax.bar(x, sv, color=C_BAR, alpha=0.6, width=0.7, zorder=2,
           label=r'$\sigma_i$')
    ax.set_ylabel(r'Singular value $\sigma_i$', fontsize=12)
    ax.set_xlabel('Component index', fontsize=12)
    ax.set_xlim(0.5, nc + 0.5)

    # Nuisance floor line
    if is_nuisance.any():
        floor = float(sv[is_nuisance].max())
        ax.axhline(floor, color=C_FLOOR, ls='--', lw=1.2, alpha=0.7,
                   zorder=4, label='Nuisance floor' if show_legend else None)

    # R² line on right axis
    ax2 = ax.twinx()
    ax2.plot(x, sum_r2, '-o', color=C_R2, lw=1.5, markersize=3,
             alpha=0.85, zorder=5,
             label=r'Per-component $R^2$' if show_legend else None)
    ax2.axhline(0, color=C_R2, ls='-', lw=0.5, alpha=0.3, zorder=1)

    r2_max   = max(abs(sum_r2.max()), abs(sum_r2.min()), 0.05)
    r2_upper = max(r2_max * 1.3, 0.05)
    r2_lower = min(-0.05, sum_r2.min() * 1.3)
    ax2.set_ylim(r2_lower, r2_upper)
    ax2.set_ylabel(r'$R^2$ (per component)', fontsize=12, color=C_R2)
    ax2.tick_params(axis='y', colors=C_R2, labelsize=10)

    # Title
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)

    # Annotation box
    n_sig = int(is_signal.sum())
    if delta_val == 0.0 and n_sig == 0:
        delta_str = r'$\hat{\Delta}_{' + delta_name + r'}$ undefined'
    else:
        delta_str = (r'$\hat{\Delta}_{' + delta_name +
                     r'}$ = ' + f'{delta_val:.2f}')
    ann = f'#signal = {n_sig},  {delta_str}'
    ax.text(0.97, 0.95, ann, transform=ax.transAxes, fontsize=10,
            va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='#999', alpha=0.9))


def main():
    # Load data
    d   = np.load(str(NPZ_PATH), allow_pickle=True)
    with open(JSON_PATH) as f:
        res = json.load(f)

    cca_r2 = d['cca_r2']
    if cca_r2.ndim == 2:
        cca_r2 = cca_r2.sum(axis=1)

    delta_ca = res['delta_ca_direct']
    delta_cp = res['delta_cp_direct']

    print(f"CCA: #signal={int(d['cca_n_signal'])}, Δ_CA={delta_ca:.4f}")
    print(f"CP:  #signal={int(d['cp_direct_n_signal'])}, Δ_CP={delta_cp:.4f}")

    # Figure setup
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    _plot_panel(
        axes[0],
        sv        = d['cca_sv'],
        sum_r2    = cca_r2,
        is_signal = d['cca_is_signal'],
        delta_val = delta_ca,
        delta_name= 'CA',
        title     = PAIR_LABEL + r': $\hat{C}$ (CCA)',
        show_legend=True,
    )
    _plot_panel(
        axes[1],
        sv        = d['cp_direct_sv'],
        sum_r2    = d['cp_direct_sum_r2'],
        is_signal = d['cp_direct_is_signal'],
        delta_val = delta_cp,
        delta_name= 'CP',
        title     = PAIR_LABEL + r': $\hat{A}$ (CP)',
        show_legend=False,
    )

    # Shared legend at bottom
    legend_elements = [
        Patch(facecolor=C_SIGNAL,   edgecolor='#999', alpha=0.6, label='Signal'),
        Patch(facecolor=C_NUISANCE, edgecolor='#999', alpha=0.6, label='Nuisance'),
        Patch(facecolor=C_BAR,      edgecolor='#999', alpha=0.6, label=r'$\sigma_i$'),
        plt.Line2D([0], [0], color=C_R2, lw=1.5, marker='o', markersize=3,
                   label=r'Per-component $R^2$'),
        plt.Line2D([0], [0], color=C_FLOOR, ls='--', lw=1.2,
                   label='Nuisance floor'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=10, frameon=True, edgecolor='#ccc',
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Signal vs nuisance spectra underlying the regime predictions',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout(rect=[0, 0.05, 1, 0.98])

    out_png = OUT_DIR / 'bmmc_phase_diagnostics.png'
    out_pdf = OUT_DIR / 'bmmc_phase_diagnostics.pdf'
    fig.savefig(str(out_png), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(str(out_pdf), bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"\nSaved: {out_png}")
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    main()
