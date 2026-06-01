"""
CaptionCOCO Training Script
============================
SSL pre-training on the image-caption CaptionCOCO dataset:

  --mode ca                  Cross-modal Alignment (VICReg on projections)
  --mode cp_strong_to_weak   Image encoder -> predict caption tokens (CE loss)
  --mode cp_weak_to_strong   Text encoder  -> predict image pixels  (MSE loss)
  --mode supervised          Supervised baseline (cross-entropy on image encoder)

Supports multi-GPU training via PyTorch DDP.
Launch with torchrun:

    torchrun --nproc_per_node=4 --standalone -m src.coco.train \\
        --mode ca \\
        --nuisance_strength 0.5 \\
        --image_noise_std 0.1 \\
        --text_dropout_prob 0.1 \\
        --save_dir checkpoints/coco/my_run

``--batch_size`` is per-GPU; effective batch = batch_size x world_size.

After pre-training, a linear probe is trained on the frozen encoder
(enc_x for CA/CP_S2W/supervised, enc_y for CP_W2S) and Top-1 / Top-5
accuracy (80 COCO classes, dominant-object label) is reported.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.coco.data import (
    CaptionCOCO, DEFAULT_COCO_ROOT, IMAGENET_MEAN, IMAGENET_STD,
    NUM_CLASSES, PAD_IDX,
)
from src.coco.models import CaptionDecoder, TextEncoder
from src.imagenet.models import (
    LinearProbe, PixelDecoder, ProjectionHead,
    ResNetEncoder, ViTEncoder,
)


# =============================================================================
# DDP helpers
# =============================================================================

def setup_ddp() -> Tuple[int, int, int]:
    """Initialize NCCL process group.  Returns (rank, local_rank, world_size)."""
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    dist.destroy_process_group()


def reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """All-reduce a scalar tensor, averaging across ranks."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()


# =============================================================================
# Loss functions
# =============================================================================

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    inv_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
) -> torch.Tensor:
    """VICReg loss for cross-modal alignment (CA mode)."""
    sim_loss = F.mse_loss(x, y)

    std_x = torch.sqrt(x.var(dim=0) + 1e-4)
    std_y = torch.sqrt(y.var(dim=0) + 1e-4)
    std_loss = torch.mean(torch.relu(1 - std_x)) + torch.mean(torch.relu(1 - std_y))

    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)
    cov_x = (x.T @ x) / (x.size(0) - 1)
    cov_y = (y.T @ y) / (y.size(0) - 1)
    cov_loss = (off_diagonal(cov_x).pow(2).sum() / x.size(1)
                + off_diagonal(cov_y).pow(2).sum() / y.size(1))

    return inv_weight * sim_loss + var_weight * std_loss + cov_weight * cov_loss


def caption_ce_loss(
    pred_logits: torch.Tensor,
    target_tokens: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> torch.Tensor:
    """Cross-entropy loss for non-autoregressive caption prediction."""
    B, L, V = pred_logits.shape
    return F.cross_entropy(
        pred_logits.reshape(B * L, V),
        target_tokens.reshape(B * L),
        ignore_index=pad_idx,
    )


def pixel_mse_loss(
    pred: torch.Tensor,
    target_normalized: torch.Tensor,
    denorm_mean: Tuple[float, ...],
    denorm_std: Tuple[float, ...],
) -> torch.Tensor:
    """MSE between decoder output (in [0,1]) and de-normalised target image."""
    mean = torch.tensor(denorm_mean, device=target_normalized.device).view(1, -1, 1, 1)
    std = torch.tensor(denorm_std, device=target_normalized.device).view(1, -1, 1, 1)
    target_raw = (target_normalized * std + mean).clamp(0, 1)
    return F.mse_loss(pred, target_raw)


# =============================================================================
# Training epoch
# =============================================================================

def train_one_epoch(
    mode: str,
    enc_x: nn.Module,
    enc_y: Optional[nn.Module],
    proj_x: Optional[nn.Module],
    proj_y: Optional[nn.Module],
    caption_decoder: Optional[nn.Module],
    pixel_decoder: Optional[nn.Module],
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
    max_grad_norm: float = 0.0,
    all_params: Optional[list] = None,
    classifier: Optional[nn.Module] = None,
) -> float:
    """Run one training epoch; returns mean loss (averaged across ranks)."""
    enc_x.train()
    if enc_y is not None:
        enc_y.train()
    for m in [proj_x, proj_y, caption_decoder, pixel_decoder, classifier]:
        if m is not None:
            m.train()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        cap = batch["caption"].to(device, non_blocking=True)

        optimizer.zero_grad()

        if mode == "ca":
            z_x = enc_x(img)
            z_y = enc_y(cap)
            loss = vicreg_loss(proj_x(z_x), proj_y(z_y))

        elif mode == "cp_strong_to_weak":
            logits = caption_decoder(enc_x(img))
            loss = caption_ce_loss(logits, cap)

        elif mode == "cp_weak_to_strong":
            pred = pixel_decoder(enc_y(cap))
            loss = pixel_mse_loss(pred, img, IMAGENET_MEAN, IMAGENET_STD)

        elif mode == "supervised":
            lbl = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            loss = F.cross_entropy(classifier(enc_x(img)), lbl)

        loss.backward()
        if max_grad_norm > 0 and all_params is not None:
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)
        optimizer.step()

        total_loss += reduce_mean(loss.detach()).item()
        n_batches += 1

    mean_loss = total_loss / max(n_batches, 1)
    if writer is not None:
        writer.add_scalar("pretrain/train_loss", mean_loss, epoch)
    return mean_loss


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(
    mode: str,
    enc_x: nn.Module,
    enc_y: Optional[nn.Module],
    proj_x: Optional[nn.Module],
    proj_y: Optional[nn.Module],
    caption_decoder: Optional[nn.Module],
    pixel_decoder: Optional[nn.Module],
    loader: DataLoader,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter],
    classifier: Optional[nn.Module] = None,
) -> float:
    """Compute validation loss (averaged across ranks)."""
    enc_x.eval()
    if enc_y is not None:
        enc_y.eval()
    for m in [proj_x, proj_y, caption_decoder, pixel_decoder, classifier]:
        if m is not None:
            m.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        cap = batch["caption"].to(device, non_blocking=True)

        if mode == "ca":
            loss = vicreg_loss(proj_x(enc_x(img)), proj_y(enc_y(cap)))
        elif mode == "cp_strong_to_weak":
            loss = caption_ce_loss(caption_decoder(enc_x(img)), cap)
        elif mode == "cp_weak_to_strong":
            loss = pixel_mse_loss(
                pixel_decoder(enc_y(cap)), img, IMAGENET_MEAN, IMAGENET_STD,
            )
        elif mode == "supervised":
            lbl = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            loss = F.cross_entropy(classifier(enc_x(img)), lbl)

        total_loss += loss.item()
        n_batches += 1

    stats = torch.tensor([total_loss, float(n_batches)], device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    val_loss = stats[0].item() / max(stats[1].item(), 1)

    if writer is not None:
        writer.add_scalar("pretrain/val_loss", val_loss, epoch)
    return val_loss


# =============================================================================
# Linear Probe evaluation
# =============================================================================

def run_linear_probe(
    encoder: nn.Module,
    latent_dim: int,
    coco_root: str,
    probe_epochs: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    save_dir: Path,
    rank: int,
    world_size: int,
    input_key: str = "image",
    vocab=None,
    max_caption_len: int = 64,
) -> dict:
    """
    Train a linear probe on frozen encoder features.

    ``input_key`` selects what to feed the encoder:
      - ``"image"`` for image encoders (CA / CP_S2W / supervised)
      - ``"caption"`` for the text encoder (CP_W2S)
    """
    if rank == 0:
        print("\n-- Linear Probe ----------------------------------------")

    probe_ds_kwargs = dict(
        coco_root=coco_root,
        nuisance_strength=0.0,
        image_noise_std=0.0,
        text_dropout_prob=0.0,
        augment=False,
        vocab=vocab,
        max_caption_len=max_caption_len,
    )
    train_ds = CaptionCOCO(split="train", **probe_ds_kwargs)
    val_ds = CaptionCOCO(split="val", **probe_ds_kwargs)

    probe_train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True,
    )
    probe_val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=probe_train_sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=probe_val_sampler,
        num_workers=num_workers, pin_memory=True,
    )

    enc = encoder.module if isinstance(encoder, DDP) else encoder
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)

    probe = LinearProbe(input_dim=latent_dim, num_classes=NUM_CLASSES).to(device)
    probe = DDP(probe, device_ids=[device.index])
    opt = torch.optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    for ep in range(1, probe_epochs + 1):
        probe_train_sampler.set_epoch(ep)
        probe.train()
        for batch in train_loader:
            inp = batch[input_key].to(device, non_blocking=True)
            lbl = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            with torch.no_grad():
                feat = enc(inp)
            loss = crit(probe(feat), lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if rank == 0 and (ep % 10 == 0 or ep == probe_epochs):
            print(f"  Probe epoch {ep}/{probe_epochs}")

    # Evaluate
    probe.eval()
    correct1 = correct5 = total = 0
    with torch.no_grad():
        for batch in val_loader:
            inp = batch[input_key].to(device, non_blocking=True)
            lbl = torch.as_tensor(batch["label"], dtype=torch.long, device=device)
            logits = probe(enc(inp))
            _, top5_pred = logits.topk(5, dim=1, largest=True, sorted=True)
            correct1 += (top5_pred[:, 0] == lbl).sum().item()
            correct5 += (top5_pred == lbl.unsqueeze(1)).any(dim=1).sum().item()
            total += lbl.size(0)

    stats = torch.tensor([correct1, correct5, total], dtype=torch.long, device=device)
    if dist.is_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    correct1, correct5, total = stats.tolist()

    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total
    results = {"top1": round(top1, 4), "top5": round(top5, 4)}

    if rank == 0:
        results_path = save_dir / "probe_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Top-1: {top1:.2f}%  |  Top-5: {top5:.2f}%")
        print(f"  Saved to {results_path}")

    return results


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CaptionCOCO pre-training (DDP)")
    # -- data --
    p.add_argument("--coco_root", type=str, default=DEFAULT_COCO_ROOT)
    p.add_argument("--nuisance_strength", type=float, default=0.0)
    p.add_argument("--nuisance_mode", type=str, default="text_only",
                   choices=["text_only", "style", "scene", "caption_swap"],
                   help="Nuisance injection mode: text_only (original), "
                        "style (visual transforms + text), "
                        "scene (scene-context from annotations), "
                        "caption_swap (within-category caption swapping).")
    p.add_argument("--style_jitter", type=float, default=0.0,
                   help="Style-mode nuisance decorrelation [0-1]. Per group, "
                        "prob of describing an independent style in the caption "
                        "vs. the one applied to the image.")
    p.add_argument("--image_style_strength", type=float, default=0.0,
                   help="Image-only style transform strength [0-1]. "
                        "Applied as weak modality noise (no text description). "
                        "Controls how many style groups are applied to the image.")
    # image noise params
    p.add_argument("--image_noise_std", type=float, default=0.0)
    p.add_argument("--image_blur_sigma", type=float, default=0.0)
    p.add_argument("--image_jpeg_quality", type=int, default=100)
    p.add_argument("--image_cutout_fraction", type=float, default=0.0)
    # text noise params
    p.add_argument("--text_dropout_prob", type=float, default=0.0)
    p.add_argument("--text_swap_prob", type=float, default=0.0)
    p.add_argument("--text_insert_prob", type=float, default=0.0)
    p.add_argument("--text_char_noise_prob", type=float, default=0.0)
    p.add_argument("--text_synonym_prob", type=float, default=0.0)
    p.add_argument("--max_caption_len", type=int, default=64)
    # -- mode / architecture --
    p.add_argument("--mode", type=str, default="ca",
                   choices=["ca", "cp_strong_to_weak", "cp_weak_to_strong",
                            "supervised"])
    p.add_argument("--backbone", type=str, default="resnet18",
                   choices=["resnet18", "resnet50", "vit_b_16", "vit_b_32",
                            "vit_s_16"])
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--proj_hidden_dim", type=int, default=512)
    p.add_argument("--text_embed_dim", type=int, default=256)
    p.add_argument("--text_num_layers", type=int, default=2)
    p.add_argument("--text_num_heads", type=int, default=4)
    # -- training --
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=256,
                   help="Per-GPU batch size.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--probe_epochs", type=int, default=30)
    p.add_argument("--warmup_epochs", type=int, default=0)
    p.add_argument("--max_grad_norm", type=float, default=0.0)
    p.add_argument("--probe_only", action="store_true")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no_pretrained", dest="pretrained", action="store_false")
    p.add_argument("--freeze_backbone", action="store_true")
    # -- output --
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--results_csv", type=str,
                   default="checkpoints/coco_results.csv")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    # -- DDP init --
    rank, local_rank, world_size = setup_ddp()
    is_main = rank == 0
    device = torch.device(f"cuda:{local_rank}")

    save_dir = Path(args.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    writer = SummaryWriter(log_dir=str(save_dir / "tb")) if is_main else None

    if is_main:
        print(f"Starting pre-training -- mode: {args.mode}  |  "
              f"world_size: {world_size}")
        print(f"  backbone={args.backbone}, latent_dim={args.latent_dim}, "
              f"nuisance={args.nuisance_strength} "
              f"(mode={args.nuisance_mode}, jitter={args.style_jitter}, "
              f"img_style={args.image_style_strength})")
        print(f"  img_noise={args.image_noise_std}, blur={args.image_blur_sigma}, "
              f"jpeg={args.image_jpeg_quality}, cutout={args.image_cutout_fraction}")
        print(f"  txt_drop={args.text_dropout_prob}, swap={args.text_swap_prob}, "
              f"insert={args.text_insert_prob}, char={args.text_char_noise_prob}, "
              f"syn={args.text_synonym_prob}")
        print(f"  per-GPU batch_size={args.batch_size}  ->  "
              f"effective batch_size={args.batch_size * world_size}")

    # -- Datasets (rank 0 builds vocab first) ------------------------------
    ds_kwargs = dict(
        coco_root=args.coco_root,
        nuisance_strength=args.nuisance_strength,
        nuisance_mode=args.nuisance_mode,
        style_jitter=args.style_jitter,
        image_style_strength=args.image_style_strength,
        image_noise_std=args.image_noise_std,
        image_blur_sigma=args.image_blur_sigma,
        image_jpeg_quality=args.image_jpeg_quality,
        image_cutout_fraction=args.image_cutout_fraction,
        text_dropout_prob=args.text_dropout_prob,
        text_swap_prob=args.text_swap_prob,
        text_insert_prob=args.text_insert_prob,
        text_char_noise_prob=args.text_char_noise_prob,
        text_synonym_prob=args.text_synonym_prob,
        max_caption_len=args.max_caption_len,
    )

    if is_main:
        train_ds = CaptionCOCO(split="train", **ds_kwargs)
    dist.barrier()
    if not is_main:
        train_ds = CaptionCOCO(split="train", **ds_kwargs)
    val_ds = CaptionCOCO(split="val", vocab=train_ds.vocab, **ds_kwargs)

    vocab_size = train_ds.vocab_size

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    # -- Models ------------------------------------------------------------
    EncoderCls = ViTEncoder if args.backbone.startswith("vit_") else ResNetEncoder

    # Image encoder (always created)
    enc_x_raw = EncoderCls(
        backbone=args.backbone, output_dim=args.latent_dim,
        in_channels=3, pretrained=args.pretrained,
    ).to(device)

    if args.freeze_backbone:
        for p in enc_x_raw.net.parameters():
            p.requires_grad_(False)
        if is_main:
            trainable = sum(p.numel() for p in enc_x_raw.parameters()
                            if p.requires_grad)
            total = sum(p.numel() for p in enc_x_raw.parameters())
            print(f"  Backbone frozen: {trainable:,}/{total:,} params trainable")

    enc_x = DDP(enc_x_raw, device_ids=[local_rank])

    # Text encoder (only for modes that need it)
    enc_y = enc_y_raw = None
    if args.mode in ("ca", "cp_weak_to_strong"):
        enc_y_raw = TextEncoder(
            vocab_size=vocab_size,
            output_dim=args.latent_dim,
            embed_dim=args.text_embed_dim,
            num_layers=args.text_num_layers,
            num_heads=args.text_num_heads,
            max_len=args.max_caption_len,
            pad_idx=PAD_IDX,
        ).to(device)
        enc_y = DDP(enc_y_raw, device_ids=[local_rank])

    def _trainable(module):
        return [p for p in module.parameters() if p.requires_grad]

    proj_x = proj_y = caption_dec = pixel_dec = classifier = None

    if args.mode == "ca":
        proj_x = DDP(
            ProjectionHead(args.latent_dim, args.proj_dim,
                           args.proj_hidden_dim).to(device),
            device_ids=[local_rank],
        )
        proj_y = DDP(
            ProjectionHead(args.latent_dim, args.proj_dim,
                           args.proj_hidden_dim).to(device),
            device_ids=[local_rank],
        )
        opt_params = (_trainable(enc_x) + _trainable(enc_y)
                      + list(proj_x.parameters()) + list(proj_y.parameters()))

    elif args.mode == "cp_strong_to_weak":
        caption_dec = DDP(
            CaptionDecoder(
                latent_dim=args.latent_dim,
                vocab_size=vocab_size,
                max_len=args.max_caption_len,
            ).to(device),
            device_ids=[local_rank],
        )
        opt_params = _trainable(enc_x) + list(caption_dec.parameters())

    elif args.mode == "cp_weak_to_strong":
        pixel_dec = DDP(
            PixelDecoder(latent_dim=args.latent_dim, out_channels=3).to(device),
            device_ids=[local_rank],
        )
        opt_params = _trainable(enc_y) + list(pixel_dec.parameters())

    elif args.mode == "supervised":
        classifier = DDP(
            LinearProbe(input_dim=args.latent_dim,
                        num_classes=NUM_CLASSES).to(device),
            device_ids=[local_rank],
        )
        opt_params = _trainable(enc_x) + list(classifier.parameters())

    optimizer = torch.optim.AdamW(opt_params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    if args.warmup_epochs > 0:
        from torch.optim.lr_scheduler import (
            CosineAnnealingLR, LinearLR, SequentialLR,
        )
        warmup = LinearLR(optimizer, start_factor=1e-4, end_factor=1.0,
                          total_iters=args.warmup_epochs)
        cosine = CosineAnnealingLR(optimizer,
                                   T_max=args.epochs - args.warmup_epochs)
        scheduler = SequentialLR(optimizer, [warmup, cosine],
                                 milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )

    # -- Resume from checkpoint --------------------------------------------
    ckpt_path = save_dir / "checkpoint.pth"
    metrics_path = save_dir / "metrics.csv"

    best_val_loss = float("inf")
    start_epoch = 1

    if is_main and not ckpt_path.exists():
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_loss"])
    if ckpt_path.exists():
        if is_main:
            print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc_x.module.load_state_dict(ckpt["enc_x"])
        if enc_y is not None and "enc_y" in ckpt:
            enc_y.module.load_state_dict(ckpt["enc_y"])
        if proj_x is not None and "proj_x" in ckpt:
            proj_x.module.load_state_dict(ckpt["proj_x"])
        if proj_y is not None and "proj_y" in ckpt:
            proj_y.module.load_state_dict(ckpt["proj_y"])
        if caption_dec is not None and "caption_decoder" in ckpt:
            caption_dec.module.load_state_dict(ckpt["caption_decoder"])
        if pixel_dec is not None and "pixel_decoder" in ckpt:
            pixel_dec.module.load_state_dict(ckpt["pixel_decoder"])
        if classifier is not None and "classifier" in ckpt:
            classifier.module.load_state_dict(ckpt["classifier"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        best_val_loss = ckpt["best_val_loss"]
        start_epoch = ckpt["epoch"] + 1
        if is_main:
            print(f"  Resuming from epoch {start_epoch}, "
                  f"best_val_loss={best_val_loss:.4f}")

    # -- Probe-only mode ---------------------------------------------------
    if args.probe_only:
        best_path = save_dir / "best_model.pth"
        if not best_path.exists():
            raise FileNotFoundError(f"--probe_only requires {best_path}")
        if is_main:
            print(f"Probe-only mode: loading {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        enc_x.module.load_state_dict(ckpt["enc_x"])
        if enc_y is not None and "enc_y" in ckpt:
            enc_y.module.load_state_dict(ckpt["enc_y"])
        _mf = save_dir / "metrics.csv"
        if _mf.exists():
            with open(_mf) as _f:
                _reader = csv.reader(_f)
                next(_reader, None)
                _vl = [float(row[2]) for row in _reader if len(row) > 2]
                if _vl:
                    best_val_loss = min(_vl)
    else:
        # -- Pre-training loop ---------------------------------------------
        for epoch in range(start_epoch, args.epochs + 1):
            train_sampler.set_epoch(epoch)
            train_loss = train_one_epoch(
                args.mode, enc_x, enc_y, proj_x, proj_y,
                caption_dec, pixel_dec,
                train_loader, optimizer, device, epoch, writer,
                max_grad_norm=args.max_grad_norm, all_params=opt_params,
                classifier=classifier,
            )
            val_loss = validate(
                args.mode, enc_x, enc_y, proj_x, proj_y,
                caption_dec, pixel_dec,
                val_loader, device, epoch, writer,
                classifier=classifier,
            )
            scheduler.step()

            if is_main:
                print(f"Epoch {epoch:3d}/{args.epochs} | "
                      f"train={train_loss:.4f} | val={val_loss:.4f}")
                with open(metrics_path, "a", newline="") as f:
                    csv.writer(f).writerow([epoch, train_loss, val_loss])

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        "enc_x": enc_x.module.state_dict(),
                    }
                    if enc_y is not None:
                        best_state["enc_y"] = enc_y.module.state_dict()
                    if classifier is not None:
                        best_state["classifier"] = \
                            classifier.module.state_dict()
                    torch.save(best_state, save_dir / "best_model.pth")

                ckpt_state = {
                    "epoch": epoch,
                    "enc_x": enc_x.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                }
                if enc_y is not None:
                    ckpt_state["enc_y"] = enc_y.module.state_dict()
                if proj_x is not None:
                    ckpt_state["proj_x"] = proj_x.module.state_dict()
                if proj_y is not None:
                    ckpt_state["proj_y"] = proj_y.module.state_dict()
                if caption_dec is not None:
                    ckpt_state["caption_decoder"] = \
                        caption_dec.module.state_dict()
                if pixel_dec is not None:
                    ckpt_state["pixel_decoder"] = \
                        pixel_dec.module.state_dict()
                if classifier is not None:
                    ckpt_state["classifier"] = \
                        classifier.module.state_dict()
                torch.save(ckpt_state, ckpt_path)

        if writer is not None:
            writer.close()
        if is_main:
            print(f"\nPre-training done. Best val_loss={best_val_loss:.4f}")

        # Barrier so non-rank-0 processes wait for rank 0 to finish saving
        dist.barrier()

        best_path = save_dir / "best_model.pth"
        if best_path.exists():
            if is_main:
                print(f"Loading best model from {best_path} for probe")
            ckpt = torch.load(best_path, map_location=device,
                              weights_only=False)
            enc_x.module.load_state_dict(ckpt["enc_x"])
            if enc_y is not None and "enc_y" in ckpt:
                enc_y.module.load_state_dict(ckpt["enc_y"])

    dist.barrier()

    # -- Linear Probe ------------------------------------------------------
    if args.mode == "cp_weak_to_strong":
        probe_encoder = enc_y
        probe_input = "caption"
    else:
        probe_encoder = enc_x
        probe_input = "image"

    probe_results = run_linear_probe(
        encoder=probe_encoder,
        latent_dim=args.latent_dim,
        coco_root=args.coco_root,
        probe_epochs=args.probe_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        save_dir=save_dir,
        rank=rank,
        world_size=world_size,
        input_key=probe_input,
        vocab=train_ds.vocab,
        max_caption_len=args.max_caption_len,
    )

    # -- Append to global results CSV (rank 0 only) ------------------------
    if is_main:
        csv_path = Path(args.results_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            cw = csv.writer(f)
            if write_header:
                cw.writerow([
                    "mode", "backbone", "latent_dim",
                    "nuisance_strength", "nuisance_mode",
                    "style_jitter", "image_style_strength",
                    "image_noise_std", "text_dropout_prob",
                    "epochs", "best_val_loss", "top1", "top5", "save_dir",
                ])
            cw.writerow([
                args.mode, args.backbone, args.latent_dim,
                args.nuisance_strength, args.nuisance_mode,
                args.style_jitter, args.image_style_strength,
                args.image_noise_std, args.text_dropout_prob,
                args.epochs, round(best_val_loss, 6),
                probe_results["top1"], probe_results["top5"],
                str(save_dir),
            ])
        print(f"Results appended to {csv_path}")

    cleanup_ddp()


if __name__ == "__main__":
    main()
