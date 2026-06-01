
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from src.dsprites.data import DSpritesMultiModal

def test_dataset():
    print("Initializing dataset...")
    # Ensure raw file exists or Mock it. 
    root = os.path.join(os.path.dirname(__file__), "..", "src", "dsprites")
    ds = DSpritesMultiModal(root=root, download=True, shared_factors=[4,5], dy=10)
    
    print(f"Dataset length: {len(ds)}")
    assert len(ds) == 737280, f"Expected 737280, got {len(ds)}"

    print("Checking item shape...")
    x, y, z = ds[0]
    print(f"Shapes: x={x.shape}, y={y.shape}, z={z.shape}")
    assert x.shape == (1, 64, 64)
    assert y.shape == (10,)
    assert z.shape == (6,)

    print("All tests passed!")

if __name__ == "__main__":
    test_dataset()
