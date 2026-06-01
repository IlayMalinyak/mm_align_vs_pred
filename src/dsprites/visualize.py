
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from src.dsprites.data import StereoDSprites

def visualize_samples(dataset, num_samples=5, save_path="stereo_dsprites_samples.png"):
    """
    Visualize random samples from the dataset.
    dataset: StereoDSprites instance
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(4 * num_samples, 8))
    
    # Ensure axes is 2D
    if num_samples == 1:
        axes = axes.reshape(2, 1)

    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        x, y, meta = dataset[idx]
        # Meta: [shape_id, posX, posY]
        
        # 1. Plot View L
        ax = axes[0, i]
        img_np = x.squeeze().numpy()
        im = ax.imshow(img_np, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f"Left (Idx: {idx})\nShape: {int(meta[0])}\nPos: ({meta[1]:.2f}, {meta[2]:.2f})")
        ax.axis('off')
        
        # 2. Plot View R
        ax = axes[1, i]
        img_np = y.squeeze().numpy()
        im = ax.imshow(img_np, cmap='gray', interpolation='nearest', vmin=0, vmax=1)
        ax.set_title(f"Right (Weak Noise)\nJit: {dataset.jitter_sigma}, Noise: {dataset.weak_noise_std}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    # Dataset params
    root = "src/dsprites"
    
    images_dir = os.path.join(os.path.dirname(__file__), "images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
        
    jitter_levels = [0.0, 0.2, 0.5]
    noise_levels = [0.2, 0.5, 0.9] # weak, medium, strong
    
    for jit in jitter_levels:
        for noise in noise_levels:
            print(f"Generating samples for jitter_sigma={jit}, weak_noise_std={noise}...")
            ds = StereoDSprites(root=images_dir, download=True, n_samples=100, 
                                jitter_sigma=jit, weak_noise_std=noise)
            
            visualize_samples(ds, num_samples=5, save_path=f"{images_dir}/stereo_dsprites_jit{jit}_noise{noise}.png")
