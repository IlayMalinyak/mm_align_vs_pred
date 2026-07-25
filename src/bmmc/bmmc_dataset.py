"""
NeurIPS 2021 BMMC CITE-seq dataset loader.

Source: GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad
        https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/

Structure (single h5ad, both modalities):
  - adata.var['feature_types'] == 'GEX'  → RNA genes
  - adata.var['feature_types'] == 'ADT'  → surface proteins
  - adata.layers['counts']               → raw counts
  - adata.X                              → log-normalized / processed
  - adata.obs['cell_type']               → signal label  (~30 populations)
  - adata.obs['DonorID']                 → noise label   (13 donors)
  - adata.obs['batch']                   → site+donor    (e.g. 's1d1')

90,261 cells × (GEX + ADT) features.

Usage:
    rna_adata, adt_adata, obs_df = load_bmmc(cache_path)
    rna_X, adt_X = rna_adata.X, adt_adata.X   # arrays for downstream use
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, Dict
import gzip
import shutil


# ── paths ───────────────────────────────────────────────────────────────
DEFAULT_CACHE_GZ  = '/home/ilay.kamai/work/citeseq/bmmc/neurips2021_cite_BMMC.h5ad.gz'
DEFAULT_CACHE_H5AD = '/home/ilay.kamai/work/citeseq/bmmc/neurips2021_cite_BMMC.h5ad'
DEFAULT_RNA_EMB   = '/home/ilay.kamai/work/citeseq/bmmc/rna_embeddings_scgpt.npy'
DEFAULT_RNA_META  = '/home/ilay.kamai/work/citeseq/bmmc/rna_meta.parquet'


def load_bmmc(
    cache_path: str = DEFAULT_CACHE_H5AD,
    cache_gz: str = DEFAULT_CACHE_GZ,
    n_top_genes: int = 2000,
    min_cells_per_type: int = 20,
) -> Tuple:
    """
    Load and split the NeurIPS 2021 BMMC CITE-seq AnnData into RNA and ADT.

    Returns:
        rna_adata   : AnnData  (n_cells × n_hvg), log-normalized RNA
        adt_adata   : AnnData  (n_cells × n_adt), CLR-normalized proteins
        obs_df      : DataFrame with cell_type, DonorID, batch columns
    """
    import anndata as ad

    h5ad_path = Path(cache_path)

    # Decompress gz if needed
    if not h5ad_path.exists():
        gz_path = Path(cache_gz)
        if not gz_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {cache_path} or {cache_gz}.\n"
                f"Download with:\n"
                f"  wget -O {cache_gz} "
                f"'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/"
                f"GSE194122/suppl/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz'"
            )
        print(f"Decompressing {gz_path} ...")
        with gzip.open(gz_path, 'rb') as f_in, open(h5ad_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print(f"  -> {h5ad_path}")

    print(f"Loading BMMC from {h5ad_path} ...")
    adata = ad.read_h5ad(h5ad_path)
    print(f"  Full: {adata.shape}  obs cols: {list(adata.obs.columns)[:10]} ...")

    # ── split by feature_types ──────────────────────────────────────────
    if 'feature_types' not in adata.var.columns:
        raise ValueError("Expected adata.var['feature_types'] with 'GEX'/'ADT' values")

    gex_mask = adata.var['feature_types'] == 'GEX'
    adt_mask = adata.var['feature_types'] == 'ADT'
    print(f"  GEX genes: {gex_mask.sum()}  ADT proteins: {adt_mask.sum()}")

    rna_full = adata[:, gex_mask].copy()
    adt_adata = adata[:, adt_mask].copy()

    # ── HVG selection for RNA ───────────────────────────────────────────
    import scanpy as sc
    rna_full.var_names_make_unique()
    sc.pp.highly_variable_genes(rna_full, n_top_genes=n_top_genes,
                                subset=True, flavor='seurat_v3',
                                layer='counts')
    rna_adata = rna_full
    print(f"  RNA after HVG ({n_top_genes}): {rna_adata.shape}")
    print(f"  ADT: {adt_adata.shape}")

    # ── filter rare cell types ──────────────────────────────────────────
    cell_type_counts = adata.obs['cell_type'].value_counts()
    valid_types = cell_type_counts[cell_type_counts >= min_cells_per_type].index
    mask = rna_adata.obs['cell_type'].isin(valid_types)
    rna_adata = rna_adata[mask].copy()
    adt_adata = adt_adata[mask].copy()
    print(f"  After filtering rare types (<{min_cells_per_type}): {rna_adata.n_obs} cells")

    # ── metadata ────────────────────────────────────────────────────────
    # batch = site+donor combo (12 unique), DonorNumber = donor person (9 unique)
    obs_df = rna_adata.obs[['cell_type', 'DonorID', 'DonorNumber', 'batch', 'Site']].copy()
    obs_df['cell_barcode'] = obs_df.index
    obs_df = obs_df.reset_index(drop=True)

    n_types  = obs_df['cell_type'].nunique()
    n_donors = obs_df['DonorNumber'].nunique()
    n_batch  = obs_df['batch'].nunique()
    print(f"  Cell types ({n_types}): {sorted(obs_df['cell_type'].unique())[:5]} ...")
    print(f"  Donors ({n_donors}): {sorted(obs_df['DonorNumber'].unique())}")
    print(f"  Batches ({n_batch}): {sorted(obs_df['batch'].unique())}")

    return rna_adata, adt_adata, obs_df


def to_dense(X) -> np.ndarray:
    if sp.issparse(X):
        return X.toarray().astype(np.float32)
    return np.array(X, dtype=np.float32)


class BMMCDataset(Dataset):
    """
    In-memory paired dataset for BMMC CITE-seq.

    Args:
        rna_embeddings : (N, D_rna) numpy array — scGPT frozen embeddings
        adt_X          : (N, D_adt) numpy array — CLR-normalized ADT
        obs_df         : DataFrame with cell_type, DonorID, batch
        indices        : optional subset indices
    """
    def __init__(
        self,
        rna_embeddings: np.ndarray,
        adt_X: np.ndarray,
        obs_df: pd.DataFrame,
        indices: Optional[np.ndarray] = None,
    ):
        if indices is not None:
            rna_embeddings = rna_embeddings[indices]
            adt_X = adt_X[indices]
            obs_df = obs_df.iloc[indices].reset_index(drop=True)

        self.rna = torch.from_numpy(rna_embeddings.astype(np.float32))
        self.adt = torch.from_numpy(adt_X.astype(np.float32))
        self.obs = obs_df.reset_index(drop=True)

        # Encode labels as integers
        self.cell_types = sorted(obs_df['cell_type'].unique())
        self.donors     = sorted(obs_df['DonorNumber'].unique())
        ct2i = {c: i for i, c in enumerate(self.cell_types)}
        dn2i = {d: i for i, d in enumerate(self.donors)}
        self.cell_type_ids = torch.tensor(
            obs_df['cell_type'].map(ct2i).values, dtype=torch.long)
        self.donor_ids = torch.tensor(
            obs_df['DonorNumber'].map(dn2i).values, dtype=torch.long)

    def __len__(self):
        return len(self.rna)

    def __getitem__(self, idx):
        return {
            'rna': self.rna[idx],
            'adt': self.adt[idx],
            'cell_type': self.cell_type_ids[idx],
            'donor': self.donor_ids[idx],
        }


def get_splits(
    rna_embeddings: np.ndarray,
    adt_X: np.ndarray,
    obs_df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    batch_size: int = 512,
    num_workers: int = 4,
) -> Dict:
    """
    Stratified split by DonorID (hold out entire donors for val/test).

    With 9 donors: ~6 train, ~1 val, ~2 test.
    """
    rng = np.random.default_rng(seed)
    donors = sorted(obs_df['DonorNumber'].unique())
    donor_arr = np.array(donors)
    rng.shuffle(donor_arr)

    n = len(donor_arr)
    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))
    train_donors = set(donor_arr[:n_train])
    val_donors   = set(donor_arr[n_train:n_train + n_val])
    test_donors  = set(donor_arr[n_train + n_val:])

    train_idx = np.where(obs_df['DonorNumber'].isin(train_donors))[0]
    val_idx   = np.where(obs_df['DonorNumber'].isin(val_donors))[0]
    test_idx  = np.where(obs_df['DonorNumber'].isin(test_donors))[0]

    print(f"Donor split — train: {sorted(train_donors)}  "
          f"val: {sorted(val_donors)}  test: {sorted(test_donors)}")
    print(f"  cells — train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    def make_loader(idx, shuffle):
        ds = BMMCDataset(rna_embeddings, adt_X, obs_df, indices=idx)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=True)

    return {
        'train': make_loader(train_idx, shuffle=True),
        'val':   make_loader(val_idx,   shuffle=False),
        'test':  make_loader(test_idx,  shuffle=False),
        'n_adt': adt_X.shape[1],
        'rna_dim': rna_embeddings.shape[1],
        'n_cell_types': obs_df['cell_type'].nunique(),
        'n_donors': obs_df['DonorNumber'].nunique(),
        'cell_types': sorted(obs_df['cell_type'].unique()),
        'donors': sorted(obs_df['DonorNumber'].unique()),
        'train_obs': obs_df.iloc[train_idx].reset_index(drop=True),
        'val_obs':   obs_df.iloc[val_idx].reset_index(drop=True),
        'test_obs':  obs_df.iloc[test_idx].reset_index(drop=True),
    }
