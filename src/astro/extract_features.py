"""
Extract features from frozen LAMOST + Kepler encoders for the full cross-matched
catalog and save to cache. Uses the same splits as cross_modal.py (seed=42).
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import DataLoader

MULTIDESA_ROOT = Path(os.environ.get('MULTIDESA_ROOT', str(Path(__file__).parent)))
sys.path.insert(0, str(MULTIDESA_ROOT))

from src.astro.cross_modal import (
    DEFAULT_CONFIG, EVAL_LABELS, CrossModalPairDataset,
    build_lamost_encoder, build_kepler_encoder,
    extract_features_batch, build_transforms,
)

SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 8
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CACHE_DIR = Path(os.environ.get('ASTRO_CACHE_DIR', str(MULTIDESA_ROOT / 'logs/cross_modal/cached_features')))


@torch.no_grad()
def extract_split(name, df_split, lamost_enc, kepler_enc, lamost_transforms, kepler_transforms):
    """Extract features for one split, return arrays."""
    ds = CrossModalPairDataset(
        df_split,
        lamost_hdf5_dir=DEFAULT_CONFIG['lamost_hdf5_dir'],
        kepler_npy_dir=DEFAULT_CONFIG['kepler_npy_dir'],
        lamost_transforms=lamost_transforms,
        kepler_transforms=kepler_transforms,
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    all_z_a, all_z_b, all_labels = [], [], []
    n_errors = 0

    for i, batch in enumerate(loader):
        z_a, z_b = extract_features_batch(lamost_enc, kepler_enc, batch, DEVICE)
        all_z_a.append(z_a.cpu().numpy())
        all_z_b.append(z_b.cpu().numpy())
        all_labels.append(batch['labels'].numpy())

        zeros = (batch['flux_lamost'].abs().sum(1) == 0).sum().item()
        zeros += (batch['flux_kepler'].abs().sum((1, 2)) == 0).sum().item()
        n_errors += zeros

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{name}] batch {i+1}/{len(loader)}")

    z_a = np.concatenate(all_z_a, axis=0)
    z_b = np.concatenate(all_z_b, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"  [{name}] done: z_a={z_a.shape}, z_b={z_b.shape}, errors={n_errors}/{len(ds)}")
    return z_a, z_b, labels


def main():
    print(f"Device: {DEVICE}")
    print(f"Cache dir: {CACHE_DIR}")

    # Load and split data (same logic as build_on_the_fly_loaders)
    print("\nLoading cross-match catalog...")
    df = pd.read_parquet(DEFAULT_CONFIG['crossmatch_parquet'])
    both = df[df['has_lamost'] & df['has_kepler']].copy()
    both = both.sort_values('lamost_snr', ascending=False).drop_duplicates('gaia_source_id')
    print(f"  {len(both)} unique paired stars")

    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(both))
    n_train = int(len(both) * 0.7)
    n_val = int(len(both) * 0.15)

    splits = {
        'train': both.iloc[idx[:n_train]],
        'val': both.iloc[idx[n_train:n_train + n_val]],
        'test': both.iloc[idx[n_train + n_val:]],
    }
    for name, df_split in splits.items():
        print(f"  {name}: {len(df_split)} samples")

    # Build encoders
    print("\nLoading frozen encoders...")
    lamost_enc = build_lamost_encoder(DEFAULT_CONFIG, DEVICE)
    kepler_enc = build_kepler_encoder(DEFAULT_CONFIG, DEVICE)

    # Build transforms
    lamost_transforms, kepler_transforms = build_transforms()

    # Extract features per split
    os.makedirs(CACHE_DIR, exist_ok=True)

    for name, df_split in splits.items():
        print(f"\nExtracting {name}...")
        z_a, z_b, labels = extract_split(
            name, df_split, lamost_enc, kepler_enc,
            lamost_transforms, kepler_transforms,
        )

        # Save metadata (labels + star IDs)
        meta = df_split[['gaia_source_id', 'KID', 'lamost_obsid', 'ra', 'dec'] + EVAL_LABELS].reset_index(drop=True)

        np.save(CACHE_DIR / f'z_lamost_{name}.npy', z_a)
        np.save(CACHE_DIR / f'z_kepler_{name}.npy', z_b)
        np.save(CACHE_DIR / f'labels_{name}.npy', labels)
        meta.to_parquet(CACHE_DIR / f'meta_{name}.parquet')
        print(f"  Saved to {CACHE_DIR}/{{z_lamost,z_kepler,labels,meta}}_{name}.*")

    # Summary
    print(f"\n{'='*60}")
    print(f"Feature extraction complete!")
    print(f"LAMOST dim: {z_a.shape[1]}, Kepler dim: {z_b.shape[1]}")
    print(f"Files saved to: {CACHE_DIR}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
