"""
Unimodal VICReg pretraining for the ADT encoder.

Two augmented views of the same cell are created via:
  - Gaussian noise  (sigma ~ Uniform(0, noise_std))
  - Random feature dropout  (p ~ Uniform(0, dropout_p))

The ADTEncoder backbone + two projection heads are trained with VICReg loss.
Only the backbone checkpoint is saved (projection heads are discarded).

Usage:
    python -m src.bmmc.pretrain_adt
    python -m src.bmmc.pretrain_adt --epochs 200 --noise_std 0.2 --dropout_p 0.2

Output:
    src/logs/bmmc/adt_pretrained/adt_encoder.pt   — backbone state dict
    src/logs/bmmc/adt_pretrained/config.json       — hyperparameters
    src/logs/bmmc/adt_pretrained/loss_curve.png    — training curve
"""
from __future__ import annotations

import sys
import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MM_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(MM_ROOT / 'src'))
sys.path.insert(0, str(MM_ROOT / 'src/bmmc'))

from cross_modal_bmmc import ADTEncoder, ProjectionHead, vicreg_loss
from bmmc_dataset import load_bmmc

BMMC_H5AD = Path('/home/ilay.kamai/work/citeseq/bmmc/neurips2021_cite_BMMC.h5ad')
OUT_DIR   = MM_ROOT / 'src/logs/bmmc/adt_pretrained'


# ── Dataset ──────────────────────────────────────────────────────────────────

class ADTAugmentDataset(Dataset):
    """Returns two independently augmented views of each cell's ADT vector."""

    def __init__(self, X: np.ndarray, noise_std: float = 0.1, dropout_p: float = 0.1):
        self.X         = torch.from_numpy(X).float()
        self.noise_std = noise_std
        self.dropout_p = dropout_p

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise with random strength
        sigma = torch.empty(1).uniform_(0, self.noise_std).item()
        x = x + sigma * torch.randn_like(x)
        # Random feature dropout
        p = torch.empty(1).uniform_(0, self.dropout_p).item()
        mask = torch.bernoulli(torch.full_like(x, 1 - p))
        x = x * mask
        return x

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        return self._augment(x), self._augment(x)


# ── Model ─────────────────────────────────────────────────────────────────────

class ADTVICReg(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], embed_dim: int,
                 proj_hidden: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = ADTEncoder(input_dim, hidden_dims, embed_dim, dropout)
        self.proj    = ProjectionHead(embed_dim, proj_hidden, proj_dim)

    def forward(self, x):
        return self.proj(self.encoder(x))


# ── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load ADT data
    print("Loading ADT data via load_bmmc()...")
    import scipy.sparse as sp
    _, adt_adata, _ = load_bmmc(str(BMMC_H5AD))
    X = adt_adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = X.astype(np.float32)
    print(f"  ADT: {X.shape}")

    input_dim = X.shape[1]

    # Small held-out split for early stopping (VICReg loss proxy, not supervised)
    dataset  = ADTAugmentDataset(X, noise_std=args.noise_std, dropout_p=args.dropout_p)
    n_val    = max(1, int(0.05 * len(dataset)))
    n_train  = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True, drop_last=False)

    # Model
    model = ADTVICReg(
        input_dim   = input_dim,
        hidden_dims = args.hidden_dims,
        embed_dim   = args.embed_dim,
        proj_hidden = args.proj_hidden,
        proj_dim    = args.proj_dim,
        dropout     = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best_val, best_epoch, patience_count = float('inf'), 0, 0
    train_losses, val_losses = [], []

    for epoch in range(1, args.epochs + 1):
        # ── train ──
        model.train()
        ep_loss = 0.0
        for v1, v2 in train_loader:
            v1, v2 = v1.to(device), v2.to(device)
            z1, z2 = model(v1), model(v2)
            loss, _ = vicreg_loss(z1, z2,
                                  sim_w=args.sim_w, var_w=args.var_w, cov_w=args.cov_w)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        ep_loss /= len(train_loader)
        train_losses.append(ep_loss)

        # ── val (SSL loss proxy for early stopping) ──
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for v1, v2 in val_loader:
                v1, v2 = v1.to(device), v2.to(device)
                z1, z2 = model(v1), model(v2)
                loss, _ = vicreg_loss(z1, z2,
                                      sim_w=args.sim_w, var_w=args.var_w, cov_w=args.cov_w)
                vl += loss.item()
        vl /= len(val_loader)
        val_losses.append(vl)
        scheduler.step()

        if vl < best_val:
            best_val, best_epoch, patience_count = vl, epoch, 0
            torch.save(model.encoder.state_dict(), OUT_DIR / 'adt_encoder.pt')
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs}  "
                  f"train={ep_loss:.4f}  val={vl:.4f}  best={best_val:.4f} (ep{best_epoch})")

        if patience_count >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    print(f"\nBest val loss: {best_val:.4f} at epoch {best_epoch}")
    print(f"Encoder saved: {OUT_DIR / 'adt_encoder.pt'}")

    # Save config
    cfg = vars(args)
    cfg['input_dim']     = int(input_dim)
    cfg['n_cells']       = int(X.shape[0])
    cfg['best_epoch']    = best_epoch
    cfg['best_val_loss'] = float(best_val)
    cfg['stopped_epoch'] = len(train_losses)
    with open(OUT_DIR / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)

    # Loss curve
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(train_losses, label='train', linewidth=1.5)
    ax.plot(val_losses,   label='val',   linewidth=1.5)
    ax.axvline(best_epoch - 1, color='gray', linestyle='--', linewidth=1, label=f'best (ep{best_epoch})')
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('VICReg loss', fontsize=13)
    ax.set_title('ADT VICReg pretraining', fontsize=14)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'loss_curve.png', dpi=150)
    print(f"Loss curve saved: {OUT_DIR / 'loss_curve.png'}")

    return model.encoder


def parse_args():
    p = argparse.ArgumentParser()
    # Architecture
    p.add_argument('--hidden_dims', type=int, nargs='+', default=[256, 512])
    p.add_argument('--embed_dim',   type=int, default=512)
    p.add_argument('--proj_hidden', type=int, default=512)
    p.add_argument('--proj_dim',    type=int, default=256)
    p.add_argument('--dropout',     type=float, default=0.1)
    # Augmentation
    p.add_argument('--noise_std',   type=float, default=0.1,
                   help='Max Gaussian noise std (sampled from Uniform(0, noise_std))')
    p.add_argument('--dropout_p',   type=float, default=0.1,
                   help='Max feature dropout probability (sampled from Uniform(0, dropout_p))')
    # VICReg weights
    p.add_argument('--sim_w',       type=float, default=25.0)
    p.add_argument('--var_w',       type=float, default=25.0)
    p.add_argument('--cov_w',       type=float, default=1.0)
    # Optimization
    p.add_argument('--epochs',      type=int,   default=200)
    p.add_argument('--patience',    type=int,   default=10)
    p.add_argument('--batch_size',  type=int,   default=1024)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--weight_decay',type=float, default=1e-4)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
