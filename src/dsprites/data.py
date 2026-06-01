
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import urllib.request
from typing import List, Optional, Tuple, Dict

class StereoDSprites(Dataset):
    """
    Stereo-dSprites dataset.
    Renders two views (Left, Right) from a canonical dSprites shape
    using stochastic camera matrices (Baseline + Jitter).
    """
    URL = "https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    FILENAME = "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

    def __init__(self, root: str, 
                 n_samples: int = 10000,
                 baseline: float = 0.4,
                 jitter_sigma: float = 0.05,
                 strong_noise_std: float = 0.1,
                 weak_noise_std: float = 0.2,
                 pos_range: float = 1.0,
                 download: bool = True):
        
        self.root = root
        self.filepath = os.path.join(root, self.FILENAME)
        if download and not os.path.exists(self.filepath):
            self._download()

        # Load dSprites
        data = np.load(self.filepath, allow_pickle=True, encoding='latin1')
        self.raw_imgs = torch.from_numpy(data['imgs']).float().unsqueeze(1) # (N, 1, 64, 64)
        self.latents_classes = torch.from_numpy(data['latents_classes']).long()
        
        # Metadata for lookup
        self.latents_sizes = data['metadata'][()]['latents_sizes']
        self.latents_bases = np.concatenate((np.cumprod(self.latents_sizes[::-1])[::-1][1:], [1,]))
        
        # Find Canonical Indices for each shape (Square=0, Ellipse=1, Heart=2)
        # Canonical: Scale=max(5), Orient=0, PosX=16, PosY=16
        self.canonical_idxs = []
        for s in range(3):
            idx = self._get_index([0, s, 5, 0, 16, 16])
            self.canonical_idxs.append(idx)
        
        self.canonical_imgs = self.raw_imgs[self.canonical_idxs] # (3, 1, 64, 64)
        
        self.n_samples = n_samples
        self.baseline = baseline
        self.jitter_sigma = jitter_sigma
        self.strong_noise_std = strong_noise_std
        self.weak_noise_std = weak_noise_std
        self.pos_range = pos_range

    def _get_index(self, latents_indices):
        return np.dot(latents_indices, self.latents_bases).astype(int)

    def _download(self):
        print(f"Downloading {self.URL} to {self.filepath}...")
        urllib.request.urlretrieve(self.URL, self.filepath)
        print("Download complete.")

    def __len__(self):
        return self.n_samples

    def _get_affine_matrix(self, tx, ty, theta):
        """
        Constructs a 2x3 affine matrix for translation and rotation.
        theta is in radians.
        """
        # torch.affine_grid expects theta for rotation
        # Matrix: [ [cos, -sin, tx], [sin, cos, ty] ]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        
        # Note: grid_sample interpretation of tx, ty is shifted? 
        # Actually, it's easier to use a tensor
        mat = torch.tensor([
            [cos_t, -sin_t, tx],
            [sin_t,  cos_t, ty]
        ], dtype=torch.float32)
        return mat

    def __getitem__(self, idx):
        # Seed RNG for deterministic generation per index
        # We need a large range for seeds to avoid correlation between idx 0 and 1 if logic is simple
        # Hash idx to get a seed? Or just use idx if n_samples < 2**32
        seed = idx + 123456789 # offset
        rng = np.random.RandomState(seed)
        
        # Step 1: Sample Latents
        shape_id = rng.randint(0, 3)
        # Scale world position sampling by pos_range
        # Default range is [-0.5, 0.5] (range 1.0)
        p_world = rng.uniform(-0.5 * self.pos_range, 0.5 * self.pos_range, size=2)
        
        canonical_img = self.canonical_imgs[shape_id].unsqueeze(0) # (1, 1, 64, 64)
        
        # Step 2 & 3: Views
        views = []
        for baseline_shift in [-self.baseline/2, self.baseline/2]:
            # Jitter
            dx, dy = rng.normal(0, self.jitter_sigma, size=2)
            dtheta = rng.normal(0, self.jitter_sigma)
            
            tx = p_world[0] + baseline_shift + dx
            ty = p_world[1] + dy
            
            # Construct Matrix
            theta = dtheta
            affine_mat = self._get_affine_matrix(-tx, -ty, -theta).unsqueeze(0)
            
            # Warp
            # Note: affine_grid is deterministic given affine_mat
            grid = F.affine_grid(affine_mat, canonical_img.size(), align_corners=False)
            view = F.grid_sample(canonical_img, grid, align_corners=False).squeeze(0)
            views.append(view)
            
        x, y = views
        
        # Step 4: Modality Noise (Add noise to X and Y)
        # Generate noise using the same rng
        noise_x = rng.normal(0, self.strong_noise_std, size=x.shape).astype(np.float32)
        x = x + torch.from_numpy(noise_x)
        x = torch.clamp(x, 0, 1)

        noise_y = rng.normal(0, self.weak_noise_std, size=y.shape).astype(np.float32)
        y = y + torch.from_numpy(noise_y)
        y = torch.clamp(y, 0, 1)
        
        # Metadata return [shape_id, posX, posY]
        return x, y, torch.tensor([shape_id, p_world[0], p_world[1]])

if __name__ == "__main__":
    # Test
    ds = StereoDSprites(root="src/dsprites", n_samples=10, pos_range=0.5, download=False)
    x, y, meta = ds[0]
    print(f"Generated Stereo Pair: {x.shape}, {y.shape}")
    print(f"Metadata: {meta}")
