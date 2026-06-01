
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
import numpy as np
from src.shapes3d.data import StereoShapes3D


def visualize_samples(dataset, num_samples=5, save_path="stereo_shapes3d_samples.png"):
    """Visualize random samples from the dataset."""
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))

    if num_samples == 1:
        axes = axes.reshape(2, 1)

    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for i, idx in enumerate(indices):
        x, y, meta = dataset[idx]

        ax = axes[0, i]
        img_np = x.permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img_np, interpolation='nearest')
        ax.set_title(f"Left (Idx: {idx})\nShape: {StereoShapes3D.SHAPE_NAMES[int(meta[0])]}\nPos: ({meta[1]:.2f}, {meta[2]:.2f})")
        ax.axis('off')

        ax = axes[1, i]
        img_np = y.permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img_np, interpolation='nearest')
        ax.set_title(f"Right (Weak Noise)\nJit: {dataset.jitter_sigma}, Noise: {dataset.weak_noise_std}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    root = "src/shapes3d"

    images_dir = os.path.join(root, "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    jitter_levels = [0.0, 0.2, 0.5]
    noise_levels = [0.2, 0.5, 0.9]

    for jit in jitter_levels:
        for noise in noise_levels:
            print(f"Generating samples for jitter_sigma={jit}, weak_noise_std={noise}...")
            ds = StereoShapes3D(root=root, download=False, n_samples=100,
                                jitter_sigma=jit, weak_noise_std=noise)

            visualize_samples(ds, num_samples=5,
                              save_path=f"{images_dir}/stereo_shapes3d_jit{jit}_noise{noise}.png")
