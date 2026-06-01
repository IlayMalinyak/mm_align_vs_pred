
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import urllib.request
import h5py


class StereoShapes3D(Dataset):
    """
    Stereo-3DShapes dataset.
    Renders two views (Left, Right) from a canonical 3DShapes image
    using stochastic camera matrices (Baseline + Jitter).

    Uses Google's 3dshapes dataset (480K RGB 64x64 images).
    Signal factor: shape (4 classes: cube, cylinder, sphere, capsule).
    Nuisance factors: floor_hue, wall_hue, object_hue, scale, orientation.
    """
    URL = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
    FILENAME = "3dshapes.h5"

    # Factor sizes: floor_hue(10), wall_hue(10), object_hue(10), scale(8), shape(4), orientation(15)
    FACTOR_SIZES = np.array([10, 10, 10, 8, 4, 15])
    SHAPE_NAMES = ['Cube', 'Cylinder', 'Sphere', 'Capsule']

    def __init__(self, root: str,
                 n_samples: int = 10000,
                 baseline: float = 0.4,
                 jitter_sigma: float = 0.05,
                 rotation_sigma: float = 0.02,
                 strong_noise_std: float = 0.1,
                 weak_noise_std: float = 0.2,
                 pos_range: float = 1.0,
                 download: bool = True):

        self.root = root
        self.filepath = os.path.join(root, self.FILENAME)
        if download and not os.path.exists(self.filepath):
            self._download()

        # Compute index bases for flat indexing into the Cartesian product
        self.latents_bases = np.concatenate(
            (np.cumprod(self.FACTOR_SIZES[::-1])[::-1][1:], [1])
        )

        # Load canonical images: one per shape class
        # Canonical factors: floor_hue=0, wall_hue=0, object_hue=0, scale=max(7), shape=s, orientation=0
        self.canonical_imgs = []
        with h5py.File(self.filepath, 'r') as f:
            images = f['images']
            for s in range(4):
                idx = self._get_index([0, 0, 0, 7, s, 0])
                img = images[idx]  # (64, 64, 3) uint8
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, 64, 64)
                self.canonical_imgs.append(img_t)
        self.canonical_imgs = torch.stack(self.canonical_imgs)  # (4, 3, 64, 64)

        self.n_samples = n_samples
        self.baseline = baseline
        self.jitter_sigma = jitter_sigma
        self.rotation_sigma = rotation_sigma
        self.strong_noise_std = strong_noise_std
        self.weak_noise_std = weak_noise_std
        self.pos_range = pos_range

    def _get_index(self, latents_indices):
        return np.dot(latents_indices, self.latents_bases).astype(int)

    def _download(self):
        os.makedirs(self.root, exist_ok=True)
        print(f"Downloading {self.URL} to {self.filepath}...")
        urllib.request.urlretrieve(self.URL, self.filepath)
        print("Download complete.")

    def __len__(self):
        return self.n_samples

    def _get_affine_matrix(self, tx, ty, theta):
        """Constructs a 2x3 affine matrix for translation and rotation."""
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        mat = torch.tensor([
            [cos_t, -sin_t, tx],
            [sin_t,  cos_t, ty]
        ], dtype=torch.float32)
        return mat

    def __getitem__(self, idx):
        seed = idx + 123456789
        rng = np.random.RandomState(seed)

        # Step 1: Sample shape and world position
        shape_id = rng.randint(0, 4)
        p_world = rng.uniform(-0.5 * self.pos_range, 0.5 * self.pos_range, size=2)

        canonical_img = self.canonical_imgs[shape_id].unsqueeze(0)  # (1, 3, 64, 64)

        # Step 2: Create stereo views with baseline + jitter
        views = []
        for baseline_shift in [-self.baseline / 2, self.baseline / 2]:
            dx, dy = rng.normal(0, self.jitter_sigma, size=2)
            dtheta = rng.normal(0, self.rotation_sigma)

            tx = p_world[0] + baseline_shift + dx
            ty = p_world[1] + dy
            theta = dtheta

            affine_mat = self._get_affine_matrix(-tx, -ty, -theta).unsqueeze(0)
            grid = F.affine_grid(affine_mat, canonical_img.size(), align_corners=False)
            view = F.grid_sample(canonical_img, grid, align_corners=False).squeeze(0)
            views.append(view)

        x, y = views

        # Step 3: Asymmetric modality noise
        noise_x = rng.normal(0, self.strong_noise_std, size=x.shape).astype(np.float32)
        x = x + torch.from_numpy(noise_x)
        x = torch.clamp(x, 0, 1)

        noise_y = rng.normal(0, self.weak_noise_std, size=y.shape).astype(np.float32)
        y = y + torch.from_numpy(noise_y)
        y = torch.clamp(y, 0, 1)

        return x, y, torch.tensor([shape_id, p_world[0], p_world[1]])


if __name__ == "__main__":
    ds = StereoShapes3D(root="src/shapes3d", n_samples=10, download=False)
    x, y, meta = ds[0]
    print(f"Generated Stereo Pair: {x.shape}, {y.shape}")
    print(f"Metadata: {meta}")
