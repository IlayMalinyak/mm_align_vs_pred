"""
Plot COCO simplified experiment: style transform strength vs accuracy.
Three points per mode (style=0.0, 0.2, 0.5), no caption swap.
Then estimate phase parameters and plot phase diagram.

Usage:
    python -m src.coco.plot_coco_simple
    python -m src.coco.plot_coco_simple --phase   # also run phase estimation
"""
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import pandas as pd

# ── Publication rcParams ────────────────────────────────────────────
matplotlib.rcParams.update({
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
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'legend.fontsize': 14,
    'legend.title_fontsize': 14,
})

# ── Data (swap=0.0, natural caption pairing) ────────────────────────
style_strengths = [0.0, 0.2, 0.5]

# Top-1 accuracy from capswap_v2 sweep (swap=0.0)
# v2 convention: text=strong (S), image=weak (W)
#   cp_strong_to_weak in code = enc_x(img) → caption_dec = I→T = W→S (wrong)
#   cp_weak_to_strong in code = enc_y(cap) → pixel_dec   = T→I = S→W (correct)
cp_ti_top1 = [46.07, 48.59, 49.33]  # CP T→I (S→W, correct): cp_weak_to_strong in code
cp_it_top1 = [56.90, 55.53, 49.39]  # CP I→T (W→S, wrong):   cp_strong_to_weak in code
ca_top1    = [47.09, 48.24, 46.03]

# Top-5
cp_ti_top5 = [77.20, 79.70, 78.43]
cp_it_top5 = [84.06, 83.29, 77.82]
ca_top5    = [75.52, 75.42, 73.02]

# ── Colors ──────────────────────────────────────────────────────────
C_CA = '#4A90C4'
C_CP_TI = '#E07B54'   # CP T→I (S→W, correct direction)
C_CP_IT = '#7B6D8E'   # CP I→T (W→S, wrong direction)

# Phase diagram colors
C_NEITHER = '#F2F0EB'
C_CA_ONLY = '#4A90C4'
C_CP_ONLY = '#E07B54'
C_BOTH    = '#7B6D8E'
C_CA_LINE = '#1A3A5C'
C_CP_LINE = '#8B3A1E'

WIN_CA  = '#1A3A5C'
WIN_CP  = '#8B3A1E'
WIN_TIE = '#888888'

OUTDIR = 'src/logs/figures/coco_simple'
os.makedirs(OUTDIR, exist_ok=True)


def plot_accuracy():
    """Image noise transforms vs Top-1 accuracy."""
    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Map style_strength → number of active distortion groups
    # n_groups = max(1, round(strength * 6)):  0.0→0, 0.2→1, 0.5→3
    n_transforms = [0, 1, 3]

    ax.plot(n_transforms, ca_top1, 'o-', color=C_CA, lw=2.5,
            markersize=13, markeredgewidth=2.5, markeredgecolor='white',
            label='CA', zorder=5)
    ax.plot(n_transforms, cp_ti_top1, 's-', color=C_CP_TI, lw=2.5,
            markersize=13, markeredgewidth=2.5, markeredgecolor='white',
            label=r'CP$_{T \to I}$', zorder=5)
    ax.plot(n_transforms, cp_it_top1, 'D-', color=C_CP_IT, lw=2.5,
            markersize=13, markeredgewidth=2.5, markeredgecolor='white',
            label=r'CP$_{I \to T}$', zorder=5)

    ax.set_xlabel(r'Image Noise Transforms ($\#$)')
    ax.set_ylabel('Top-1 Accuracy (%)')
    ax.set_title('COCO Image–Caption')
    ax.set_xticks(n_transforms)
    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(40, 62)

    leg = ax.legend(loc='upper right', frameon=True, edgecolor='#cccccc',
                    fancybox=False, framealpha=0.95)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    ax.grid(True, alpha=0.3, linewidth=0.8)

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'coco_style_accuracy.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
# Phase estimation using raw features
# ═══════════════════════════════════════════════════════════════════════

def load_raw_features(image_style_strength, coco_root, n_samples=4952,
                      downsample=64, pca_dim=100):
    """
    Extract raw features for COCO val with given image_style_strength.

    X = downsampled image pixels → PCA
    Y = bag-of-words token counts → PCA
    """
    import torch
    import torch.nn.functional as F
    from sklearn.decomposition import PCA
    from torch.utils.data import DataLoader
    from src.coco.data import CaptionCOCO

    cache_dir = os.path.join('checkpoints', 'coco_raw_features')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(
        cache_dir,
        f"simple_style{image_style_strength}_n{n_samples}_ds{downsample}_pca{pca_dim}.npz"
    )

    if os.path.exists(cache_file):
        print(f"  Loading cached: {cache_file}")
        data = np.load(cache_file)
        X_pca, Y_pca, labels_int = data["X_pca"], data["Y_pca"], data["labels_int"]
        C = len(np.unique(labels_int))
        labels_onehot = np.zeros((len(labels_int), C))
        labels_onehot[np.arange(len(labels_int)), labels_int] = 1.0
        return X_pca, Y_pca, labels_int, labels_onehot

    # No caption swap, no text noise, no image noise — only style transform
    ds = CaptionCOCO(
        split="val", coco_root=coco_root,
        nuisance_mode="caption_swap",
        nuisance_strength=0.0,
        style_jitter=0.0,
        image_style_strength=image_style_strength,
        augment=False, seed_augmentation=True,
        image_noise_std=0.0, text_dropout_prob=0.0,
        image_blur_sigma=0.0, image_jpeg_quality=100,
        image_cutout_fraction=0.0, text_swap_prob=0.0,
        text_insert_prob=0.0, text_char_noise_prob=0.0,
    )
    loader = DataLoader(ds, batch_size=200, num_workers=0, shuffle=False)

    vocab_size = len(ds.vocab)
    img_list, bow_list, label_list = [], [], []
    count = 0

    for batch in loader:
        if count >= n_samples:
            break
        img = batch["image"]
        img_ds = F.interpolate(img, size=downsample, mode='bilinear',
                               align_corners=False)
        img_list.append(img_ds.numpy().reshape(len(img), -1))

        cap = batch["caption"].numpy()
        bow = np.zeros((len(cap), vocab_size), dtype=np.float32)
        for i, tokens in enumerate(cap):
            for t in tokens:
                if t > 0:
                    bow[i, t] += 1.0
        bow_list.append(bow)
        label_list.append(np.array(batch["label"]))
        count += img.size(0)

    X = np.concatenate(img_list)[:n_samples]
    Y = np.concatenate(bow_list)[:n_samples]
    labels_int = np.concatenate(label_list)[:n_samples]
    del img_list, bow_list, label_list, ds, loader
    gc.collect()

    print(f"  Raw dims: X={X.shape[1]}, Y={Y.shape[1]} → PCA({pca_dim})")

    pca_x = min(pca_dim, X.shape[1], X.shape[0] - 1)
    pca_y = min(pca_dim, Y.shape[1], Y.shape[0] - 1)
    X_pca = PCA(n_components=pca_x).fit_transform(X); del X
    Y_pca = PCA(n_components=pca_y).fit_transform(Y); del Y
    gc.collect()

    # Remap labels to contiguous 0..C-1
    unique_labels = np.unique(labels_int)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels_int = np.array([label_map[l] for l in labels_int])
    C = len(unique_labels)
    labels_onehot = np.zeros((len(labels_int), C))
    labels_onehot[np.arange(len(labels_int)), labels_int] = 1.0

    np.savez_compressed(cache_file, X_pca=X_pca, Y_pca=Y_pca, labels_int=labels_int)
    print(f"  Cached: {cache_file}")

    return X_pca, Y_pca, labels_int, labels_onehot


def run_phase_estimation(coco_root):
    """Estimate phase parameters for 3 style strengths."""
    from src.analyze_phase_diagram import estimate_parameters

    results = []
    for ss in style_strengths:
        print(f"\n--- style_strength={ss} ---")
        X, Y, _, labels_oh = load_raw_features(ss, coco_root)
        nc = min(20, X.shape[1], Y.shape[1])
        res = estimate_parameters(X, Y, labels_oh,
                                  n_components=nc, n_permutations=10, cv=3)
        # One row per classification method
        for mname, mres in res.get('methods', {}).items():
            results.append({
                'style_strength': ss,
                'method': mname,
                'kappa_hat': mres['kappa_hat'],
                'nu_hat': mres['nu_hat'],
                'gamma_x_hat': mres['gamma_x_hat'],
                'gamma_y_hat': mres['gamma_y_hat'],
                'gamma_tilde_x_hat': mres['gamma_tilde_x_hat'],
                'gamma_tilde_y_hat': mres['gamma_tilde_y_hat'],
                'Delta_CA_hat': mres['Delta_CA_hat'],
                'Delta_CP_hat': mres['Delta_CP_hat'],
                'predicted_regime': mres['predicted_regime'],
                'Q_x': mres.get('Q_x', float('nan')),
                'Q_y': mres.get('Q_y', float('nan')),
            })
        del X, Y, labels_oh
        gc.collect()

    df = pd.DataFrame(results)
    from src.analyze_phase_diagram import quantile_suffix as _qs
    csv_path = os.path.join(OUTDIR, f'phase_estimates{_qs()}.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(df.to_string(index=False))
    return df


def plot_phase_diagram(df):
    """Plot phase diagram with single estimated point (style=0.0), log-scale κ."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch, FancyBboxPatch
    from matplotlib.lines import Line2D

    # Use style=0.0 estimates
    row = df[df['style_strength'] == 0.0].iloc[0]
    gamma_x = row['gamma_x_hat']
    gamma_y = row['gamma_y_hat']
    gamma_tilde_y = row['gamma_tilde_y_hat']
    kappa_hat = row['kappa_hat']
    nu_hat = row['nu_hat']

    # Log-scale κ range — wide enough to show full phase structure
    kappa_min, kappa_max = 1e-2, 1e2
    n_pts = 500

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Phase regions on log-spaced κ grid
    kk_log = np.logspace(np.log10(kappa_min), np.log10(kappa_max), n_pts)
    nn = np.linspace(0, 1.0, n_pts)
    KK, NN = np.meshgrid(kk_log, nn)
    k2 = KK ** 2

    with np.errstate(divide='ignore', invalid='ignore'):
        denom_ca = np.sqrt((k2 + gamma_x) * (k2 + gamma_y))
        Delta_CA = np.where(NN > 0, k2 / (NN * denom_ca), 0)
        if gamma_tilde_y > 0:
            Delta_CP = np.where(
                NN > 0,
                k2 / (NN * np.sqrt(gamma_tilde_y) * np.sqrt(k2 + gamma_x)),
                0,
            )
        else:
            Delta_CP = np.full_like(Delta_CA, np.inf)

    regime = np.zeros_like(Delta_CA, dtype=int)
    regime[(Delta_CA > 1) & (Delta_CP <= 1)] = 1   # CA only
    regime[(Delta_CA <= 1) & (Delta_CP > 1)] = 2   # CP only
    regime[(Delta_CA > 1) & (Delta_CP > 1)] = 3    # Both

    cmap = ListedColormap([C_NEITHER, C_CA_ONLY, C_CP_ONLY, C_BOTH])
    ax.pcolormesh(KK, NN, regime, cmap=cmap, shading='auto',
                  rasterized=True)
    ax.set_xscale('log')

    # Boundary curves
    kk_fine = np.logspace(np.log10(kappa_min), np.log10(kappa_max), 800)
    k2f = kk_fine ** 2
    nu_CA = k2f / np.sqrt((k2f + gamma_x) * np.maximum(k2f + gamma_y, 1e-30))
    if gamma_tilde_y > 0:
        nu_CP = k2f / (np.sqrt(gamma_tilde_y) * np.sqrt(k2f + gamma_x))
    else:
        nu_CP = np.full_like(kk_fine, 10.0)

    ax.plot(kk_fine, np.clip(nu_CA, 0, 1), color=C_CA_LINE, lw=2.8, zorder=5)
    ax.plot(kk_fine, np.clip(nu_CP, 0, 1), color=C_CP_LINE, lw=2.8,
            ls='--', dashes=(6, 3), zorder=5)

    # Data point — star marker at (κ̂, ν̂)
    # Place at small positive ν so visible above axis
    nu_plot = max(nu_hat, 0.02)
    ax.plot(kappa_hat, nu_plot, '*', color='red', markersize=22,
            markeredgewidth=1.5, markeredgecolor='black', zorder=10)

    # Annotation
    ax.annotate(
        'COCO',
        xy=(kappa_hat, nu_plot),
        xytext=(kappa_hat * 2.5, nu_plot + 0.12),
        fontsize=16, fontweight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                  edgecolor='red', linewidth=1.5),
        arrowprops=dict(arrowstyle='-', color='black', lw=1.5),
        zorder=11,
    )

    # γ parameter box (upper left)
    gamma_text = (
        f"$\\gamma_x = {gamma_x:.0f}$\n"
        f"$\\gamma_y = {gamma_y:.2f}$\n"
        f"$\\tilde{{\\gamma}}_y = {gamma_tilde_y:.2f}$"
    )
    ax.text(0.03, 0.97, gamma_text, transform=ax.transAxes,
            fontsize=14, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#999', linewidth=1.0, alpha=0.95))

    ax.set_xlabel(r'Signal strength $\hat{\kappa}$')
    ax.set_ylabel(r'Nuisance correlation $\hat{\nu}$')
    ax.set_title('COCO Phase Diagram')
    ax.set_xlim(kappa_min, kappa_max)
    ax.set_ylim(0, 1.0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.4)

    # Legend (lower right, single column)
    legend_elements = [
        Patch(facecolor=C_NEITHER, edgecolor='#999', lw=1.0, label='Neither'),
        Patch(facecolor=C_CA_ONLY, edgecolor='#999', lw=1.0, label='CA only'),
        Patch(facecolor=C_CP_ONLY, edgecolor='#999', lw=1.0, label='CP only'),
        Patch(facecolor=C_BOTH, edgecolor='#999', lw=1.0, label='Both'),
        Line2D([0], [0], color=C_CA_LINE, lw=2.8, ls='-',
               label=r'$\Delta_{\mathrm{CA}}=1$'),
        Line2D([0], [0], color=C_CP_LINE, lw=2.8, ls='--',
               label=r'$\Delta_{\mathrm{CP}}=1$'),
    ]
    leg = ax.legend(handles=legend_elements, loc='lower right',
                    frameon=True, edgecolor='#999', fancybox=False,
                    framealpha=0.95, fontsize=13)
    for text in leg.get_texts():
        text.set_fontweight('bold')

    fig.tight_layout()
    path = os.path.join(OUTDIR, 'coco_phase_diagram.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    print(f"Saved: {path}")
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', action='store_true',
                        help='Also run phase estimation and plot phase diagram')
    parser.add_argument('--coco_root', type=str,
                        default=os.environ.get('COCO_ROOT', '/path/to/coco2017'))
    args = parser.parse_args()

    plot_accuracy()

    if args.phase:
        from src.analyze_phase_diagram import quantile_suffix as _qs
        csv_path = os.path.join(OUTDIR, f'phase_estimates{_qs()}.csv')
        if os.path.exists(csv_path):
            print(f"Loading existing estimates from {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            df = run_phase_estimation(args.coco_root)
        plot_phase_diagram(df)
