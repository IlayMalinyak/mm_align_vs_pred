"""
Generate an animated GIF sweeping gamma_tilde_y (nuisance variance in target),
showing the variance trap: CP region shrinks as nuisance variance grows.

Output: figs/phase_diagram_sweep.gif

Usage:
    python -m src.linear.make_phase_gif
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import imageio.v2 as imageio
import tempfile

from src.linear.phase_diagram import (
    make_phase_diagram,
    C_NEITHER, C_CA_ONLY, C_CP_ONLY, C_BOTH, C_CA_LINE, C_CP_LINE,
)

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'font.size': 16,
    'axes.labelsize': 20,
    'axes.titlesize': 20,
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.linewidth': 1.4,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'figs')
OUT_PATH = os.path.join(OUT_DIR, 'phase_diagram_sweep.gif')

# Fixed parameters
GAMMA_X   = 1.0
KAPPA_MAX = 3.0

# Sweep: both gamma_y and gamma_tilde_y from low to high and back
# gamma_y controls the CA boundary; gamma_tilde_y controls the CP boundary
N_FRAMES   = 60
GY_VALUES  = np.concatenate([
    np.linspace(0.05, 2.0, N_FRAMES // 2),   # forward: CA region shrinks
    np.linspace(2.0, 0.05, N_FRAMES // 2),   # reverse
])
GTY_VALUES = np.concatenate([
    np.linspace(0.05, 5.0, N_FRAMES // 2),   # forward: CP region shrinks
    np.linspace(5.0, 0.05, N_FRAMES // 2),   # reverse
])

LEGEND_ELEMENTS = [
    Patch(facecolor=C_NEITHER, edgecolor='#999', lw=1.0, label='Neither'),
    Patch(facecolor=C_CA_ONLY, edgecolor='#999', lw=1.0, label='CA only'),
    Patch(facecolor=C_CP_ONLY, edgecolor='#999', lw=1.0, label='CP only'),
    Patch(facecolor=C_BOTH,    edgecolor='#999', lw=1.0, label='Both'),
    plt.Line2D([0], [0], color=C_CA_LINE, lw=2.5, ls='-',  label=r'$\Delta_\mathrm{CA}=1$'),
    plt.Line2D([0], [0], color=C_CP_LINE, lw=2.5, ls='--', label=r'$\Delta_\mathrm{CP}=1$'),
]


def render_frame(gy: float, gty: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6, 5))

    make_phase_diagram(ax, GAMMA_X, gy, gty, kappa_max=KAPPA_MAX)

    ax.set_title(
        r'$\gamma_x={:.1f},\; \gamma_y={:.2f},\; \tilde{{\gamma}}_y={:.2f}$'.format(
            GAMMA_X, gy, gty),
        pad=8,
    )

    ax.legend(
        handles=LEGEND_ELEMENTS,
        loc='upper right', ncol=2,
        fontsize=11, frameon=True, edgecolor='#ccc',
        fancybox=False,
        handlelength=1.6, handletextpad=0.4, columnspacing=1.0,
        prop={'weight': 'bold'},
    )

    plt.tight_layout()

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    img = buf[:, :, :3].copy()
    plt.close(fig)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames = []
    for i, (gy, gty) in enumerate(zip(GY_VALUES, GTY_VALUES)):
        print(f"  Frame {i+1}/{len(GTY_VALUES)}  gamma_y={gy:.3f}  gamma_tilde_y={gty:.3f}", end='\r')
        frames.append(render_frame(gy, gty))
    print()

    # Hold at start and end
    hold = [frames[0]] * 10 + frames + [frames[-1]] * 10

    imageio.mimsave(OUT_PATH, hold, fps=15, loop=0)
    print(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
