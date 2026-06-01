import sys
import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.dsprites.data import StereoDSprites
from src.dsprites.models import ImageEncoder, ProjectionHead

try:
    import umap
except ImportError:
    print("Please install umap-learn directly using pip install umap-learn")
    sys.exit(1)

def process_experiment(exp_name, base_ckpt_dir, save_dir, data_root, device):
    """Generate UMAP for a single experiment."""
    exp_dir = os.path.join(base_ckpt_dir, exp_name)
    print(f"Loading best_model.pth from {exp_dir}")

    # Parse args from directory name
    parts = exp_name.split('_')
    
    # Simple parse based on naming convention: mode_sampX_jitY_noiseZ_[posW_]sS
    mode = parts[0]
    if mode == 'cp' and 'wrong' in exp_name:
        mode = 'cp_wrong'
        
    # Extract params for title
    # Format typically: ca_samp50000_jit0.0_noise0.5_s3
    jitter_val = "Unknown"
    noise_val = "Unknown"
    for part in parts:
        if part.startswith('jit'):
            jitter_val = part[3:]
        elif part.startswith('noise'):
            noise_val = part[5:]

    # Initialize models
    enc_x = ImageEncoder(output_dim=128).to(device)
    enc_y = ImageEncoder(output_dim=128).to(device)
    
    if mode != 'cp_wrong' and mode != 'cp':
        proj_x = ProjectionHead(128, 32).to(device)
    
    ckpt_path = os.path.join(exp_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        print(f"Error: {ckpt_path} not found. Skipping.")
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

    # Create dataset (we just need a canonical split, 5000 samples is enough for a good UMAP)
    dataset = StereoDSprites(
        root=data_root, 
        n_samples=5000, 
        download=True
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    print(f"Extracting features for {exp_name}...")
    features = []
    shapes = []
    pos_x = []
    pos_y = []

    with torch.no_grad():
        for x, y, latents in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Predict based on mode
            if mode == 'cp_wrong':
                z = enc_y(y) 
            else:
                z = enc_x(x)
                
            features.append(z.cpu().numpy())
            shapes.append(latents[:, 0].numpy())
            pos_x.append(latents[:, 1].numpy())
            pos_y.append(latents[:, 2].numpy())

    features = np.concatenate(features)
    shapes = np.concatenate(shapes)
    pos_x = np.concatenate(pos_x)
    pos_y = np.concatenate(pos_y)

    print("Generating UMAP...")
    reducer = umap.UMAP()
    emb = reducer.fit_transform(features)

    # Plot
    pos_min, pos_max = pos_x.min(), pos_x.max()
    pos_norm = (pos_x - pos_min) / (pos_max - pos_min)
    pos_scaled = 0.2 + 0.8 * pos_norm # range [0.2, 1.0]

    plt.figure(figsize=(10, 8))
    
    cmaps = {
        0: plt.cm.Blues,
        1: plt.cm.Greens,
        2: plt.cm.Reds
    }
    
    shape_labels = {
        0: 'Square',
        1: 'Ellipse',
        2: 'Heart'
    }

    for shape_id in [0, 1, 2]:
        mask = (shapes == shape_id)
        
        # Get exact colors from the map
        colors = cmaps[shape_id](pos_scaled[mask])
        
        plt.scatter(
            emb[mask, 0], 
            emb[mask, 1], 
            c=colors, 
            s=10, 
            alpha=0.8,
            label=shape_labels[shape_id]
        )

    method_str = {"ca": "CA", "cp": "CP", "cp_wrong": "CP (Wrong)", "deepcca": "CA (DeepCCA)"}.get(mode, mode.upper())
    plt.title(f"{method_str} | Jitter {jitter_val} | Weak Noise {noise_val}", fontweight='bold', fontsize=16)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    
    # Legend just for the shapes
    plt.legend(title='Shape', frameon=True, fontsize=12, title_fontsize=12)
    plt.grid(True, ls='--', alpha=0.3)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'umap_{exp_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    
    print(f"Saved advanced UMAP to {save_path}\n")

    # Clean up memory to avoid OOM in loops
    del features, shapes, pos_x, pos_y, emb, dataset, loader
    del enc_x, enc_y
    if 'proj_x' in locals(): del proj_x
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    import gc; gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_ckpt_dir', type=str, default='checkpoints', help="Base directory containing experiments")
    parser.add_argument('--save_dir', type=str, default='src/logs/figures/dsprites', help="Where to save visualisations")
    parser.add_argument('--data_root', type=str, default='src/dsprites')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    experiments = [
        "ca_samp50000_jit0.0_noise0.5_s1",
        "ca_samp50000_jit0.5_noise0.5_s1",
        "cp_samp50000_jit0.0_noise0.5_s1",
        "cp_samp50000_jit0.5_noise0.5_s1",
        "ca_samp50000_jit0.0_noise0.5_s2",
        "ca_samp50000_jit0.5_noise0.5_s2",
        "cp_samp50000_jit0.0_noise0.5_s2",
        "cp_samp50000_jit0.5_noise0.5_s2",
        "ca_samp50000_jit0.0_noise0.5_s3",
        "ca_samp50000_jit0.5_noise0.5_s3",
        "cp_samp50000_jit0.0_noise0.5_s3",
        "cp_samp50000_jit0.5_noise0.5_s3",
        "ca_samp50000_jit0.0_noise0.5_s4",
        "ca_samp50000_jit0.5_noise0.5_s4",
        "cp_samp50000_jit0.0_noise0.5_s4",
        "cp_samp50000_jit0.5_noise0.5_s4"
    ]
    
    for exp in experiments:
        process_experiment(exp, args.base_ckpt_dir, args.save_dir, args.data_root, device)

if __name__ == "__main__":
    main()
