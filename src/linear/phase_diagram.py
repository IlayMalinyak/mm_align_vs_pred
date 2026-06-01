"""
Phase diagram: CA vs CP recovery regions (publication-ready).

Axes: (κ, ν) — signal strength vs nuisance cross-modal correlation.
Regions: Neither, CA only, CP only, Both.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# ---------- style (publication-ready, matches CLAUDE.md spec) ----------
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 22,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.4,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.pad': 5,
    'ytick.major.pad': 5,
})

# ---------- colors (muted, colorblind-friendly) ----------
C_NEITHER = '#F2F0EB'
C_CA_ONLY = '#4A90C4'
C_CP_ONLY = '#E07B54'
C_BOTH    = '#7B6D8E'
C_CA_LINE = '#1A3A5C'
C_CP_LINE = '#8B3A1E'

# ---------- recovery boundaries ----------

def ca_boundary(kappa, gamma_x, gamma_y):
    return kappa**2 / np.sqrt((kappa**2 + gamma_x) * (kappa**2 + gamma_y))

def cp_boundary(kappa, gamma_x, gamma_tilde_y):
    return kappa**2 / (np.sqrt(kappa**2 + gamma_x) * np.sqrt(gamma_tilde_y))


def make_phase_diagram(ax, gamma_x, gamma_y, gamma_tilde_y,
                       kappa_max=5.0, res=1000, label=None, title=None):
    kappa = np.linspace(0, kappa_max, res)
    nu = np.linspace(0, 1.0, res)
    K, N = np.meshgrid(kappa, nu)

    ca_thresh = K**2 / np.sqrt((K**2 + gamma_x) * (K**2 + gamma_y))
    cp_thresh = K**2 / (np.sqrt(K**2 + gamma_x) * np.sqrt(gamma_tilde_y))

    ca_ok = N < ca_thresh
    cp_ok = N < cp_thresh

    region = np.zeros_like(K, dtype=int)
    region[ca_ok & ~cp_ok] = 1
    region[~ca_ok & cp_ok] = 2
    region[ca_ok & cp_ok] = 3

    cmap = ListedColormap([C_NEITHER, C_CA_ONLY, C_CP_ONLY, C_BOTH])
    ax.imshow(region, origin='lower', aspect='auto',
              extent=[0, kappa_max, 0, 1.0], cmap=cmap, vmin=0, vmax=3,
              interpolation='bilinear')

    # Boundaries (high-res for smooth curves)
    kk = np.linspace(0.01, kappa_max, 800)
    nu_ca = ca_boundary(kk, gamma_x, gamma_y)
    nu_cp = np.clip(cp_boundary(kk, gamma_x, gamma_tilde_y), 0, 1.0)

    ax.plot(kk, nu_ca, color=C_CA_LINE, ls='-', lw=2.8, zorder=5)
    ax.plot(kk, nu_cp, color=C_CP_LINE, ls='--', lw=2.8, dashes=(6, 3), zorder=5)

    ax.set_xlabel(r'Signal strength $\kappa$')
    ax.set_ylabel(r'Nuisance correlation $\nu$')
    ax.set_xlim(0, kappa_max)
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # Panel label (outside, top-left)
    if label:
        ax.text(-0.02, 1.04, label, transform=ax.transAxes,
                fontsize=22, fontweight='bold', va='bottom', ha='right')

    if title:
        ax.set_title(title, pad=10)

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)


# ---------- main figure ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Asymmetric: γ_x=1.0 (strong), γ_y=0.05 (weak); panels differ only in γ̃_y
configs = [
    # (gamma_x, gamma_y, gamma_tilde_y, kappa_max, label, title)
    (1.0, 0.05, 5.0, 3.0, '(a)',
     'Large target nuisance\n' + r'($\gamma_x=1,\; \gamma_y=0.05,\; \tilde{\gamma}_y = 5$)'),
    (1.0, 0.05, 0.05, 3.0, '(b)',
     'Small target nuisance\n' + r'($\gamma_x=1,\; \gamma_y=0.05,\; \tilde{\gamma}_y = 0.05$)'),
]

for ax, (gx, gy, gty, km, lbl, ttl) in zip(axes, configs):
    make_phase_diagram(ax, gx, gy, gty, kappa_max=km, label=lbl, title=ttl)

axes[1].set_ylabel('')

legend_elements = [
    Patch(facecolor=C_NEITHER, edgecolor='#999', lw=1.0, label='Neither'),
    Patch(facecolor=C_CA_ONLY, edgecolor='#999', lw=1.0, label='CA only'),
    Patch(facecolor=C_CP_ONLY, edgecolor='#999', lw=1.0, label='CP only'),
    Patch(facecolor=C_BOTH,    edgecolor='#999', lw=1.0, label='Both'),
    plt.Line2D([0], [0], color=C_CA_LINE, lw=2.8, ls='-',
               label='$\\Delta_{\\mathrm{CA}}=1$'),
    plt.Line2D([0], [0], color=C_CP_LINE, lw=2.8, ls='--',
               label='$\\Delta_{\\mathrm{CP}}=1$'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=6,
           fontsize=14, frameon=True, edgecolor='#ccc',
           fancybox=False, bbox_to_anchor=(0.5, -0.04),
           handlelength=2.0, handletextpad=0.5, columnspacing=1.8,
           prop={'weight': 'bold'})

plt.tight_layout(w_pad=1.5)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'logs', 'figures', 'linear')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'phase_diagram.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor='white', pad_inches=0.15)
print(f"Saved: {out_path}")
