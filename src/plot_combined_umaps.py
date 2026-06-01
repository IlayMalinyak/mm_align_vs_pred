"""
Combined UMAP Figures: 2x2 grids for dSprites and Shapes3D
===========================================================
Columns: Aligned nuisance (jitter=0) | Misaligned nuisance (jitter=0.5)
Rows:    CA (contrastive)            | CP (predictive)

Can work from:
  - Saved .npz embedding files (preferred, fast)
  - Model checkpoints (slower, runs UMAP)
  - Existing PNG images (fallback, composes as sub-images)

Usage:
    python -m src.plot_combined_umaps --dataset dsprites
    python -m src.plot_combined_umaps --dataset shapes3d
    python -m src.plot_combined_umaps --dataset shapes3d --from_checkpoints
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import seaborn as sns


# ── Config per dataset ────────────────────────────────────────────────────

DATASETS = {
    "dsprites": {
        "png_dir": "src/logs/figures/dsprites",
        "npz_dir": "src/logs/figures/dsprites",
        "ckpt_dir": "checkpoints",
        "files": {
            (0, 0): "umap_ca_samp50000_jit0.0_noise0.5_s3",
            (0, 1): "umap_ca_samp50000_jit0.5_noise0.5_s3",
            (1, 0): "umap_cp_samp50000_jit0.0_noise0.5_s3",
            (1, 1): "umap_cp_samp50000_jit0.5_noise0.5_s3",
        },
        "shape_labels": {0: "Square", 1: "Ellipse", 2: "Heart"},
        "cmaps": {0: plt.cm.Blues, 1: plt.cm.Greens, 2: plt.cm.Reds},
        "n_shapes": 3,
        "save_name": "combined_umaps_dsprites.png",
    },
    "shapes3d": {
        "png_dir": "src/logs/figures/shapes3d",
        "npz_dir": "src/logs/figures/shapes3d",
        "ckpt_dir": "checkpoints/shapes3d",
        "files": {
            (0, 0): "umap_ca_samp100000_jit0.0_noise0.5_s1",
            (0, 1): "umap_ca_samp100000_jit0.5_noise0.5_s1",
            (1, 0): "umap_cp_samp100000_jit0.0_noise0.5_s1",
            (1, 1): "umap_cp_samp100000_jit0.5_noise0.5_s1",
        },
        "shape_labels": {0: "Cube", 1: "Cylinder", 2: "Sphere", 3: "Capsule"},
        "cmaps": {0: plt.cm.Blues, 1: plt.cm.Greens, 2: plt.cm.Reds, 3: plt.cm.Purples},
        "n_shapes": 4,
        "save_name": "combined_umaps_shapes3d.png",
    },
}

ROW_LABELS = ["CA (Contrastive)", "CP (Predictive)"]
COL_LABELS = [
    "Aligned Nuisance — CP Favorable",
    "Misaligned Nuisance — CA Favorable",
]

# 1x4 layout: group by method, then by regime
PANEL_ORDER_1x4 = [
    ((0, 0), "CA — Aligned Nuisance"),
    ((0, 1), "CA — Misaligned Nuisance"),
    ((1, 0), "CP — Aligned Nuisance"),
    ((1, 1), "CP — Misaligned Nuisance"),
]


# ── Extract embeddings from checkpoints ───────────────────────────────────

def extract_and_save_embeddings(dataset_key, cfg):
    """Run model inference + UMAP, save .npz files for later reuse."""
    import torch
    from torch.utils.data import DataLoader

    try:
        import umap
    except ImportError:
        print("Please install umap-learn: pip install umap-learn")
        sys.exit(1)

    if dataset_key == "dsprites":
        from src.dsprites.data import StereoDSprites as DatasetClass
        from src.dsprites.models import ImageEncoder, ProjectionHead
        data_root = "src/dsprites"
    else:
        from src.shapes3d.data import StereoShapes3D as DatasetClass
        from src.shapes3d.models import ImageEncoder, ProjectionHead
        data_root = "./src/shapes3d"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for (row, col), base_name in cfg["files"].items():
        npz_path = os.path.join(cfg["npz_dir"], base_name + ".npz")
        if os.path.exists(npz_path):
            print(f"  {npz_path} already exists, skipping.")
            continue

        # Parse experiment name (strip "umap_" prefix)
        exp_name = base_name[5:]  # remove "umap_"
        ckpt_path = os.path.join(cfg["ckpt_dir"], exp_name, "best_model.pth")
        if not os.path.exists(ckpt_path):
            print(f"  Checkpoint not found: {ckpt_path}, skipping.")
            continue

        # Determine mode
        mode = "ca" if exp_name.startswith("ca_") else "cp"

        print(f"  Loading {ckpt_path}...")
        enc_x = ImageEncoder(output_dim=128).to(device)
        enc_y = ImageEncoder(output_dim=128).to(device)

        checkpoint = torch.load(ckpt_path, map_location=device)
        enc_x.load_state_dict(checkpoint["enc_x"])
        enc_y.load_state_dict(checkpoint["enc_y"])
        enc_x.eval()
        enc_y.eval()

        dataset = DatasetClass(root=data_root, n_samples=5000, download=True)
        loader = DataLoader(dataset, batch_size=128, shuffle=False)

        features, shapes, pos_xs = [], [], []
        with torch.no_grad():
            for x, y, latents in loader:
                x, y = x.to(device), y.to(device)
                z = enc_x(x) if mode == "ca" else enc_y(y) if mode == "cp_wrong" else enc_x(x)
                features.append(z.cpu().numpy())
                shapes.append(latents[:, 0].numpy())
                pos_xs.append(latents[:, 1].numpy())

        features = np.concatenate(features)
        shapes = np.concatenate(shapes)
        pos_x = np.concatenate(pos_xs)

        print(f"  Running UMAP for {exp_name}...")
        reducer = umap.UMAP(random_state=42)
        emb = reducer.fit_transform(features)

        os.makedirs(cfg["npz_dir"], exist_ok=True)
        np.savez_compressed(npz_path, emb=emb, shapes=shapes, pos_x=pos_x)
        print(f"  Saved embeddings: {npz_path}")

        del features, shapes, pos_x, emb, enc_x, enc_y, dataset, loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc; gc.collect()


# ── Plot from .npz embeddings ─────────────────────────────────────────────

def plot_combined_from_npz(cfg, save_dir):
    """Create 2x2 combined UMAP figure from saved .npz embeddings."""
    sns.set_theme(style="white", context="paper")

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    for (row, col), base_name in cfg["files"].items():
        npz_path = os.path.join(cfg["npz_dir"], base_name + ".npz")
        if not os.path.exists(npz_path):
            print(f"  Missing: {npz_path}")
            return False

        data = np.load(npz_path)
        emb, shapes, pos_x = data["emb"], data["shapes"], data["pos_x"]

        ax = axes[row, col]

        # Normalize position to intensity
        pmin, pmax = pos_x.min(), pos_x.max()
        if pmax - pmin > 1e-8:
            pos_norm = (pos_x - pmin) / (pmax - pmin)
        else:
            pos_norm = np.full_like(pos_x, 0.5)
        pos_scaled = 0.2 + 0.8 * pos_norm

        for shape_id in range(cfg["n_shapes"]):
            mask = shapes == shape_id
            colors = cfg["cmaps"][shape_id](pos_scaled[mask])
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=colors, s=8, alpha=0.8,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("0.7")

    # Column headers
    for col, label in enumerate(COL_LABELS):
        axes[0, col].set_title(label, fontsize=14, fontweight="bold", pad=12)

    # Row labels
    for row, label in enumerate(ROW_LABELS):
        axes[row, 0].set_ylabel(label, fontsize=14, fontweight="bold", labelpad=10)

    # ── Unified legend ────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    shape_labels = cfg["shape_labels"]
    legend_handles = []
    for shape_id in range(cfg["n_shapes"]):
        color = cfg["cmaps"][shape_id](0.6)  # representative mid-intensity
        legend_handles.append(
            Line2D([], [], marker="o", color="none", markerfacecolor=color,
                   markersize=10, label=shape_labels[shape_id])
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=cfg["n_shapes"],
        frameon=True,
        edgecolor="0.8",
        fontsize=13,
        title="Shape Class",
        title_fontsize=14,
        columnspacing=2.0,
        handletextpad=0.5,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.12, wspace=0.08)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, cfg["save_name"])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return True


# ── Fallback: compose from existing PNGs (1x4 row layout) ────────────────

def plot_combined_from_pngs(cfg, save_dir):
    """Compose 1x4 row from existing UMAP PNG files."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    for idx, ((row, col), title) in enumerate(PANEL_ORDER_1x4):
        base_name = cfg["files"][(row, col)]
        png_path = os.path.join(cfg["png_dir"], base_name + ".png")
        if not os.path.exists(png_path):
            print(f"  Missing: {png_path}")
            return False

        img = mpimg.imread(png_path)

        # Crop: top ~9% (title), bottom ~12% (x-axis label), left ~8% (y-axis label)
        h, w = img.shape[:2]
        crop_top = int(h * 0.09)
        crop_bot = int(h * 0.12)
        crop_left = int(w * 0.08)
        img = img[crop_top:h - crop_bot, crop_left:, :]

        ax = axes[idx]
        ax.imshow(img, aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=20, fontweight="bold", pad=14)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("0.7")

    # ── Unified legend ────────────────────────────────────────────────
    from matplotlib.lines import Line2D
    shape_labels = cfg["shape_labels"]
    legend_handles = []
    for shape_id in range(cfg["n_shapes"]):
        color = cfg["cmaps"][shape_id](0.6)
        legend_handles.append(
            Line2D([], [], marker="o", color="none", markerfacecolor=color,
                   markersize=14, label=shape_labels[shape_id])
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=cfg["n_shapes"],
        frameon=True,
        edgecolor="0.8",
        fontsize=18,
        title="Shape Class",
        title_fontsize=20,
        columnspacing=2.5,
        handletextpad=0.5,
    )

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.05)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, cfg["save_name"])
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["dsprites", "shapes3d", "both"], default="both")
    p.add_argument("--from_checkpoints", action="store_true",
                   help="Extract embeddings from model checkpoints and save .npz files")
    p.add_argument("--save_dir", default="src/logs/figures/")
    return p.parse_args()


def process_dataset(dataset_key, args):
    cfg = DATASETS[dataset_key]
    print(f"\n{'='*50}")
    print(f"  {dataset_key.upper()}")
    print(f"{'='*50}")

    if args.from_checkpoints:
        extract_and_save_embeddings(dataset_key, cfg)

    # Try .npz first, fall back to PNG composition
    if not plot_combined_from_npz(cfg, args.save_dir):
        print("  NPZ files not available, composing from existing PNGs...")
        if not plot_combined_from_pngs(cfg, args.save_dir):
            print(f"  ERROR: Could not create combined UMAP for {dataset_key}")


def main():
    args = parse_args()

    datasets = ["dsprites", "shapes3d"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        process_dataset(ds, args)


if __name__ == "__main__":
    main()
