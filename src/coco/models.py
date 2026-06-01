"""
COCO Experiment Models
======================
Text encoder and caption decoder for the image-caption multimodal setup.

Image encoders (ResNetEncoder, ViTEncoder), ProjectionHead, LinearProbe,
and PixelDecoder are imported from ``src.imagenet.models``.
"""

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """
    Transformer-based text encoder: token IDs -> fixed-dim vector.

    Architecture
    ------------
    Token embedding + positional embedding -> Transformer encoder
    -> mean-pool (over non-padded positions) -> linear projection.

    Parameters
    ----------
    vocab_size : Vocabulary size (including special tokens).
    output_dim : Dimensionality of the output vector.
    embed_dim  : Internal embedding / transformer dimension.
    num_layers : Number of Transformer encoder layers.
    num_heads  : Number of attention heads.
    max_len    : Maximum sequence length.
    pad_idx    : Index of the <pad> token (masked in attention).
    """

    def __init__(
        self,
        vocab_size: int,
        output_dim: int = 128,
        embed_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        max_len: int = 64,
        pad_idx: int = 0,
    ) -> None:
        super().__init__()
        self.pad_idx = pad_idx

        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_embed = nn.Embedding(max_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )
        self.fc = nn.Linear(embed_dim, output_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        tokens : LongTensor [B, L]

        Returns
        -------
        Tensor [B, output_dim]
        """
        B, L = tokens.shape
        positions = torch.arange(L, device=tokens.device)
        x = self.token_embed(tokens) + self.pos_embed(positions).unsqueeze(0)

        pad_mask = tokens == self.pad_idx  # True where padded
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Mean-pool over non-padded positions
        valid = (~pad_mask).unsqueeze(-1).float()  # [B, L, 1]
        x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        return self.fc(x)


class CaptionDecoder(nn.Module):
    """
    Non-autoregressive caption predictor: latent vector -> per-position logits.

    Each position receives the projected latent vector **plus** a learned
    positional embedding; a shared MLP then predicts the token at that
    position.  Trained with cross-entropy (ignoring ``<pad>``).

    Parameters
    ----------
    latent_dim : Dimensionality of the input latent vector.
    vocab_size : Output vocabulary size.
    max_len    : Sequence length (must match dataset ``max_caption_len``).
    hidden_dim : Width of the internal MLP.
    """

    def __init__(
        self,
        latent_dim: int,
        vocab_size: int,
        max_len: int = 64,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.project = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Embedding(max_len, hidden_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        z : Tensor [B, latent_dim]

        Returns
        -------
        Tensor [B, max_len, vocab_size]  (unnormalised logits)
        """
        B = z.size(0)
        h = self.project(z).unsqueeze(1)  # [B, 1, hidden_dim]
        pos = self.pos_embed(
            torch.arange(self.max_len, device=z.device),
        )  # [max_len, hidden_dim]
        h = h + pos.unsqueeze(0)  # [B, max_len, hidden_dim]
        return self.mlp(h)
