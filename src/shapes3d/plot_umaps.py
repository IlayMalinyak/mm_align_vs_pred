import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.shapes3d.data import StereoShapes3D
from src.shapes3d.models import ImageEncoder, ProjectionHead

try:
    import umap
except ImportError:
    print("Please install umap-learn: pip install umap-learn")
    sys.exit(1)


def process_experiment(exp_name, base_ckpt_dir, save_dir, data_root, device):
    """Generate UMAP for a single experiment."""
    exp_dir = os.path.join(base_ckpt_dir, exp_name)
    print(f"Loading best_model.pth from {exp_dir}")

    # Parse mode from directory name
    parts = exp_name.split('_')
    mode = parts[0]
    if mode == 'cp' and 'wrong' in exp_name:
        mode = 'cp_wrong'

    # Extract params for title
    jitter_val = "?"
    noise_val = "?"
    for part in parts:
        if part.startswith('jit'):
            jitter_val = part[3:]
        elif part.startswith('noise'):
            noise_val = part[5:]

    # Initialize models
    enc_x = ImageEncoder(output_dim=128).to(device)
    enc_y = ImageEncoder(output_dim=128).to(device)

    if mode not in ('cp', 'cp_wrong'):
        proj_x = ProjectionHead(128, 32).to(device)

    ckpt_path = os.path.join(exp_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print(f"  {ckpt_path} not found. Skipping.")
        return

    checkpoint = torch.load(ckpt_path, map_location=device)
    enc_x.load_state_dict(checkpoint['enc_x'])
    enc_y.load_state_dict(checkpoint['enc_y'])
    if 'proj_x' in checkpoint:
        proj_x.load_state_dict(checkpoint['proj_x'])

    enc_x.eval()
    enc_y.eval()
    if 'proj_x' in checkpoint:
        proj_x.eval()

    # 5000 samples is enough for a good UMAP
    dataset = StereoShapes3D(root=data_root, n_samples=5000, download=True)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    print(f"  Extracting features for {exp_name}...")
    features = []
    shapes = []
    pos_x_list = []
    pos_y_list = []

    with torch.no_grad():
        for x, y, latents in loader:
            x = x.to(device)
            y = y.to(device)

            if mode == 'cp_wrong':
                z = enc_y(y)
            else:
                z = enc_x(x)

            features.append(z.cpu().numpy())
            shapes.append(latents[:, 0].numpy())
            pos_x_list.append(latents[:, 1].numpy())
            pos_y_list.append(latents[:, 2].numpy())

    features = np.concatenate(features)
    shapes = np.concatenate(shapes)
    pos_x = np.concatenate(pos_x_list)
    pos_y = np.concatenate(pos_y_list)

    print("  Running UMAP...")
    reducer = umap.UMAP()
    emb = reducer.fit_transform(features)

    # Normalize position to color intensity range [0.2, 1.0]
    pos_min, pos_max = pos_x.min(), pos_x.max()
    if pos_max - pos_min > 1e-8:
        pos_norm = (pos_x - pos_min) / (pos_max - pos_min)
    else:
        pos_norm = np.full_like(pos_x, 0.5)
    pos_scaled = 0.2 + 0.8 * pos_norm

    # Plot: shape → colormap, position → intensity
    plt.figure(figsize=(10, 8))

    cmaps = {
        0: plt.cm.Blues,
        1: plt.cm.Greens,
        2: plt.cm.Reds,
        3: plt.cm.Purples,
    }

    shape_labels = {
        0: 'Cube',
        1: 'Cylinder',
        2: 'Sphere',
        3: 'Capsule',
    }

    for shape_id in range(4):
        mask = (shapes == shape_id)
        colors = cmaps[shape_id](pos_scaled[mask])
        plt.scatter(
            emb[mask, 0],
            emb[mask, 1],
            c=colors,
            s=10,
            alpha=0.8,
            label=shape_labels[shape_id],
        )

    method_str = {"ca": "CA", "cp": "CP", "cp_wrong": "CP (Wrong)", "deepcca": "CA (DeepCCA)"}.get(mode, mode.upper())
    plt.title(f"{method_str} | Jitter {jitter_val} | Weak Noise {noise_val}",
              fontweight='bold', fontsize=16)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(title='Shape', frameon=True, fontsize=12, title_fontsize=12)
    plt.grid(True, ls='--', alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'umap_{exp_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')

    print(f"  Saved UMAP to {save_path}\n")

    # Clean up
    del features, shapes, pos_x, pos_y, emb, dataset, loader
    del enc_x, enc_y
    if 'proj_x' in locals():
        del proj_x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc; gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_ckpt_dir', type=str,
                        default='checkpoints/shapes3d')
    parser.add_argument('--save_dir', type=str,
                        default='src/logs/figures/shapes3d')
    parser.add_argument('--data_root', type=str,
                        default='./src/shapes3d')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                        help="Specific experiment names to plot. If not given, auto-discovers all.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.experiments:
        experiments = args.experiments
    else:
        # Auto-discover all experiment dirs that contain best_model.pth
        experiments = []
        if os.path.isdir(args.base_ckpt_dir):
            for name in sorted(os.listdir(args.base_ckpt_dir)):
                ckpt = os.path.join(args.base_ckpt_dir, name, 'best_model.pth')
                if os.path.isfile(ckpt):
                    experiments.append(name)
        print(f"Found {len(experiments)} experiments with checkpoints.")

    for exp in experiments:
        process_experiment(exp, args.base_ckpt_dir, args.save_dir,
                           args.data_root, device)


if __name__ == "__main__":
    main()
