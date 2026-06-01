from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F 
import math
from typing import Optional, Tuple
from nn.Modules.mhsa_pro import MHA_rotary, RotaryEmbedding
from nn.models import ACFDecoder

class ConvBlock(nn.Module):
    def __init__(self, args, num_layer) -> None:
        super().__init__()
        self.activation = nn.ReLU()
        in_channels = args.encoder_dims[num_layer-1] if num_layer < len(args.encoder_dims) else args.encoder_dims[-1]
        out_channels = args.encoder_dims[num_layer] if num_layer < len(args.encoder_dims) else args.encoder_dims[-1]
        
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=args.kernel_size,
                    stride=1, padding='same', bias=False),
            nn.BatchNorm1d(num_features=out_channels),
            self.activation,
        )
        
        # Skip connection projection layer (if input/output dimensions don't match)
        self.skip_connection = None
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Apply main convolution layers
        out = self.layers(x)
        
        # Apply skip connection projection if needed
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        
        # Add skip connection
        out = out + identity
        
        return out

class CNNEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        print("Using CNN encoder with activation: ", args.activation, 'args avg_output: ', args.avg_output)
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            from nn.models import Sine
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
                kernel_size=3, out_channels = args.encoder_dims[0], stride=1, padding = 'same', bias = False),
                        nn.BatchNorm1d(args.encoder_dims[0]),
                        self.activation,
        )
        self.in_channels = args.in_channels 
        self.layers = nn.ModuleList([ConvBlock(args, i+1)
        for i in range(args.num_layers)])
        self.pool = nn.MaxPool1d(2)
        self.output_dim = args.encoder_dims[-1]
        self.min_seq_len = 2 
        self.avg_output = args.avg_output
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape)==2:
            x = x.unsqueeze(1)
        if len(x.shape)==3 and x.shape[-1]==1:
            x = x.permute(0,2,1)
        
        x = self.embedding(x.float())
        
        for m in self.layers:
            # Store input for potential skip connection across pooling
            pre_pool = x
            x = m(x)
            
            # Apply pooling if sequence length allows
            if x.shape[-1] > self.min_seq_len:
                x = self.pool(x)
        
        if self.avg_output:
            x = x.mean(dim=-1)
        else:
            x = x.permute(0,2,1)
        return x


class RoPETransformerBlock(nn.Module):
    """Transformer block with RoPE positional encoding"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Multi-head attention with RoPE
        self.attention = MHA_rotary(args)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(args.encoder_dim, args.encoder_dim * 4),
            nn.GELU(),
            nn.Linear(args.encoder_dim * 4, args.encoder_dim),
            nn.Dropout(args.dropout_p)
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(args.encoder_dim)
        self.ln2 = nn.LayerNorm(args.encoder_dim)
        
        # Dropout
        self.dropout = nn.Dropout(args.dropout_p)
        
    def forward(self, x, RoPE, key_padding_mask=None):
        # Pre-norm architecture
        # Self-attention with residual connection
        attn_out = self.attention(self.ln1(x), RoPE, key_padding_mask)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_out)
        
        return x


class RoPETransformer(nn.Module):
    """Transformer encoder with RoPE positional encoding"""
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Model dimensions
        self.encoder_dim = args.encoder_dim
        self.num_layers = args.num_layers
        self.num_heads = args.num_heads
        self.dropout_p = args.dropout_p
        
        # RoPE setup
        self.head_size = args.encoder_dim // args.num_heads
        self.rotary_ndims = int(self.head_size * 0.5)
        self.pe = RotaryEmbedding(self.rotary_ndims)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            RoPETransformerBlock(args) for _ in range(self.num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(args.encoder_dim)
        
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x: Input tensor of shape [B, T, C]
            key_padding_mask: Optional mask for padding tokens [B, T]
        
        Returns:
            Output tensor of shape [B, T, C]
        """
        B, T, C = x.size()
        
        # Generate RoPE embeddings
        RoPE = self.pe(x, seq_len=T).nan_to_num(0)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, RoPE, key_padding_mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        return x


class RoPETransformerConfig:
    """Configuration class for RoPE Transformer"""
    
    def __init__(self, 
                 encoder_dim=512,
                 num_layers=6,
                 num_heads=8,
                 dropout_p=0.1,
                 timeshift=False):
        self.encoder_dim = encoder_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.timeshift = timeshift
        
        # Validate that encoder_dim is divisible by num_heads
        assert encoder_dim % num_heads == 0, f"encoder_dim ({encoder_dim}) must be divisible by num_heads ({num_heads})"


class ViT(nn.Module):
    def __init__(self, args, transformer_args):
        super().__init__()
    # Extract key parameters
        self.seq_len = args.seq_len if hasattr(args, 'seq_len') else 1024
        self.patch_size = args.patch_size if hasattr(args, 'patch_size') else 16
        self.in_channels = args.in_channels
        self.dim = transformer_args.encoder_dim
        self.meta_dim = args.meta_dim
        self.depth = args.num_layers
        self.heads = transformer_args.num_heads
        self.num_patches = self.seq_len // self.patch_size
        
        
        # Activation
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            from nn.models import Sine
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        # Patchify and embed
        self.patch_embed = nn.Conv1d(
            self.in_channels, 
            self.dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        
        # Custom RoPE Transformer
        self.transformer = RoPETransformer(transformer_args)
        
        # meta encoder - assume scalars so input dim is 1 and meta_dim is number of meta variables
        self.meta_encoder = nn.Sequential(
            nn.Linear(1, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            self.activation,
            nn.Linear(self.dim // 2, self.dim),
        )
    
    def forward(self, x, meta=None):
        B = x.shape[0]
        
        # Handle input shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1, -2)
            
        # Patchify: [B, C, L] -> [B, num_patches, dim]
        x = self.patch_embed(x.float())  # [B, dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, dim]
        
        if meta is not None:
            assert meta.shape[1] == self.meta_dim, 'inconsistent meta shape'
            meta_embed = self.meta_encoder(meta)  # [B, meta_dim, dim]
            x = torch.cat([meta_embed, x], dim=1) # [B, num_patches + meta_dim, dim]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches + meta_dim+1, dim]

        # Transformer encoding with RoPE
        x_enc = self.transformer(x)

        return x_enc



class SpectralViT2(ViT):
    """SpectralViT using custom RoPE Transformer"""
    
    def __init__(self, args, transformer_args):
        super().__init__(args, transformer_args)
        
        # Extract key parameters
        
        self.output_dim = args.output_dim
        self.num_quantiles = args.num_quantiles
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(self.dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            self.activation,
            nn.Linear(self.dim // 2, self.dim // 4),
            nn.LayerNorm(self.dim // 4),
            self.activation,
            nn.Linear(self.dim // 4, self.patch_size * self.in_channels),

        )
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim // 2),
            nn.BatchNorm1d(self.dim // 2),
            self.activation,
            nn.Dropout(transformer_args.dropout_p),
            nn.Linear(self.dim // 2, self.output_dim * self.num_quantiles)
        )
        
        
    def forward(self, x, y=None, meta=None):
        B = x.shape[0]
           
        x_enc = super().forward(x, meta=meta)
        
        # Split cls token and patches
        cls_output = x_enc[:, 0]  # [B, dim]
        patch_output = x_enc[:, 1:]  # [B, num_patches+meta_dim, dim]
        
        # Regression from cls token
        output_reg = self.regressor(cls_output)

        if meta is not None:
            meta_output = patch_output[:, :self.meta_dim]  # [B, meta_dim, dim]
            patch_output = patch_output[:, self.meta_dim:]  # [B, num_patches, dim]
        # Reconstruction from patches
        x_dec = self.decoder(patch_output)
        
        # Reshape to original size
        x_dec = x_dec.reshape(B, self.num_patches, self.patch_size, self.in_channels)
        x_dec = x_dec.permute(0, 3, 1, 2)  # [B, C, num_patches, patch_size]
        output_dec = x_dec.reshape(B, self.in_channels, self.num_patches * self.patch_size)
        
        # Ensure correct length
        if output_dec.shape[-1] != self.seq_len:
            import torch.nn.functional as F
            output_dec = F.interpolate(output_dec, size=self.seq_len, mode='linear', align_corners=False)
        
        if self.in_channels == 1:
            output_dec = output_dec.squeeze(1)
        
        return output_reg, output_dec, cls_output



class ContrastiveViT(ViT):
    """Contrastive ViT using SimCLR/SimSiam-like framework"""
    
    def __init__(self, args, transformer_args, num_embeddings=2):
        super().__init__(args, transformer_args)

        self.output_dim = args.output_dim
        self.num_quantiles = args.num_quantiles
        
        # Extract additional parameters
        self.num_classes = args.num_classes if hasattr(args, 'num_classes') else 10
        self.projection_dim = args.projection_dim if hasattr(args, 'projection_dim') else 128
        self.temperature = args.temperature if hasattr(args, 'temperature') else 0.1
        self.use_predictor = args.use_predictor if hasattr(args, 'use_predictor') else True  # For SimSiam
        
        # Projection head for contrastive learning (maps cls_token to projection space)
        self.projector = nn.Sequential(
            nn.Linear(self.dim, self.dim),
            nn.LayerNorm(self.dim),
            self.activation,
            nn.Linear(self.dim, self.dim // 2),
            nn.LayerNorm(self.dim // 2),
            self.activation,
            nn.Linear(self.dim // 2, self.projection_dim)
        )
        
        # Predictor head (for SimSiam - predicts one view from another)
        if self.use_predictor:
            self.predictor = nn.Sequential(
                nn.Linear(self.projection_dim, self.projection_dim // 2),
                nn.BatchNorm1d(self.projection_dim // 2),
                self.activation,
                nn.Linear(self.projection_dim // 2, self.projection_dim)
            )
        
        # Regressor head (takes concatenated embeddings)
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.dim * num_embeddings),  # 2x because we concatenate two embeddings
            nn.Linear(self.dim * num_embeddings, self.dim),
            nn.BatchNorm1d(self.dim),
            self.activation,
            nn.Dropout(transformer_args.dropout_p),
            nn.Linear(self.dim, self.dim // 2),
            nn.BatchNorm1d(self.dim // 2),
            self.activation,
            nn.Linear(self.dim // 2, self.output_dim * self.num_quantiles)
        )
        
        # Classifier head (takes concatenated embeddings)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.dim * num_embeddings),  # 2x because we concatenate two embeddings
            nn.Linear(self.dim * num_embeddings, self.dim),
            nn.BatchNorm1d(self.dim),
            self.activation,
            nn.Dropout(transformer_args.dropout_p),
            nn.Linear(self.dim, self.dim // 2),
            nn.BatchNorm1d(self.dim // 2),
            self.activation,
            nn.Linear(self.dim // 2, self.num_classes)
        )

        if hasattr(args, 'acf_length') and args.acf_length > 0:
            self.acf_decoder = ACFDecoder(
                input_dim=self.output_dim,
                acf_length=args.acf_length,
                hidden_dims=getattr(args, 'acf_decoder_dims', [512, 256, 128])
            )
        else:
            self.acf_decoder = None
        
    def encode_single(self, x, meta=None):
        """Encode a single input through the ViT encoder"""
        return super().forward(x, meta=meta)
    
    def contrastive_loss(self, z1, z2):
        """
        Compute contrastive loss (InfoNCE/SimCLR style)
        z1, z2: [B, projection_dim] - projected embeddings
        """
        B = z1.shape[0]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate embeddings
        z = torch.cat([z1, z2], dim=0)  # [2B, projection_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2B, 2B]
        
        # Create labels - positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
        
        # Remove diagonal elements (self-similarity)
        mask = torch.eye(2*B, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def simsiam_loss(self, p1, p2, z1, z2):
        """
        Compute SimSiam loss (negative cosine similarity)
        p1, p2: predictions [B, projection_dim]
        z1, z2: projections [B, projection_dim] (detached)
        """
        # Normalize
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        z1 = F.normalize(z1.detach(), dim=1)  # Stop gradient
        z2 = F.normalize(z2.detach(), dim=1)  # Stop gradient
        
        # Negative cosine similarity
        loss = -(F.cosine_similarity(p1, z2).mean() + F.cosine_similarity(p2, z1).mean()) * 0.5
        
        return loss
    
    def forward(self, x1, x2, meta=None):
        """
        Forward pass with two input signals
        x1, x2: [B, C, L] or [B, L] - input signals
        meta1, meta2: [B, meta_dim] - metadata (optional)
        """
        B = x1.shape[0]
        
        # Encode both inputs independently
        x1_enc = self.encode_single(x1, meta=meta)  # [B, num_patches+meta_dim+1, dim]
        x2_enc = self.encode_single(x2, meta=meta)  # [B, num_patches+meta_dim+1, dim]
        
        # Extract cls tokens
        cls1 = x1_enc[:, 0]  # [B, dim]
        cls2 = x2_enc[:, 0]  # [B, dim]
        
        # Project to contrastive space without meta and cls tokens
        z1 = self.projector(x1_enc[:, self.meta_dim + 1:].sum(1))  # [B, projection_dim]
        z2 = self.projector(x2_enc[:, self.meta_dim + 1:].sum(1))  # [B, projection_dim]
        
        # Concatenate embeddings for downstream tasks
        combined_emb = torch.cat([cls1, cls2], dim=1)  # [B, 2*dim]
        
        # Downstream tasks
        regression_output = self.regressor(combined_emb)  # [B, output_dim]
        classification_output = self.classifier(combined_emb)  # [B, num_classes]
        
        # Prepare return dictionary
        outputs = {
            'regression': regression_output,
            'classification': classification_output,
            'embeddings': (x1_enc.mean(1), x2_enc.mean(1)),
            'combined_embedding': combined_emb,
            'projections': (z1, z2)
        }
        
        # Compute contrastive loss
        if self.use_predictor:
            # SimSiam style
            p1 = self.predictor(z1)  # [B, projection_dim]
            p2 = self.predictor(z2)  # [B, projection_dim]
            contrastive_loss = self.simsiam_loss(p1, p2, z1, z2)
            outputs['predictions'] = (p1, p2)
        else:
            # SimCLR style
            contrastive_loss = self.contrastive_loss(z1, z2)
        
        outputs['contrastive_loss'] = contrastive_loss

        if self.acf_decoder is not None:
            out['acf_reconstruction'] = self.acf_decoder(z)
        
        return outputs


class ACFContrastiveViT(ContrastiveViT):
    def __init__(self, args, transformer_args, cnn_args):
        super().__init__(args, transformer_args, num_embeddings=4)
        self.acf_encoder = CNNEncoder(cnn_args)

    def forward(self, x1, x2, meta=None):
        """
        Forward pass with two input signals
        x1, x2: [B, C, L] or [B, L] - input signals
        meta1, meta2: [B, meta_dim] - metadata (optional)
        """
        B = x1.shape[0]
        x1, acf1 = x1[:,0,:], x1[:,1,:]
        x2, acf2 = x2[:,0,:], x2[:,1,:]
        
        # Encode both inputs independently
        x1_enc = self.encode_single(x1, meta=meta)  # [B, num_patches+meta_dim+1, dim]
        x2_enc = self.encode_single(x2, meta=meta)  # [B, num_patches+meta_dim+1, dim]

        acf1_enc = self.acf_encoder(acf1)
        acf2_enc = self.acf_encoder(acf2)
        
        # Extract CLS tokens
        cls1 = x1_enc[:, 0]  # [B, dim]
        cls2 = x2_enc[:, 0]  # [B, dim]
        
         # Project to contrastive space without meta and cls tokens
        z1 = self.projector(x1_enc[:, self.meta_dim + 1:].sum(1))  # [B, projection_dim]
        z2 = self.projector(x2_enc[:, self.meta_dim + 1:].sum(1))  # [B, projection_dim]
        
        # Concatenate embeddings for downstream tasks
        combined_emb = torch.cat([cls1, cls2, acf1_enc, acf2_enc], dim=1)  # [B, 4*dim]
        
        # Downstream tasks
        regression_output = self.regressor(combined_emb)  # [B, output_dim]
        classification_output = self.classifier(combined_emb)  # [B, num_classes]
        
        # Prepare return dictionary
        outputs = {
            'regression': regression_output,
            'classification': classification_output,
            'embeddings': (x1_enc.mean(1), x2_enc.mean(1)),
            'combined_embedding': combined_emb,
            'projections': (z1, z2)
        }
        
        # Compute contrastive loss
        if self.use_predictor:
            # SimSiam style
            p1 = self.predictor(z1)  # [B, projection_dim]
            p2 = self.predictor(z2)  # [B, projection_dim]
            contrastive_loss = self.simsiam_loss(p1, p2, z1, z2)
            outputs['predictions'] = (p1, p2)
        else:
            # SimCLR style
            contrastive_loss = self.contrastive_loss(z1, z2)
        
        outputs['contrastive_loss'] = contrastive_loss
        
        return outputs

# ==========================
# Option 1: Vision Transformer Style for Spectra
# ==========================

class SpectralViT(nn.Module):
    """ViT-style architecture treating spectra as 1D images with patches"""
    
    def __init__(self, args, transformer_args):
        super().__init__()
        
        # Extract key parameters
        self.seq_len = args.seq_len if hasattr(args, 'seq_len') else 1024
        self.patch_size = args.patch_size if hasattr(args, 'patch_size') else 16
        self.in_channels = args.in_channels
        self.dim = transformer_args.encoder_dim
        self.depth = args.num_layers
        self.heads = transformer_args.num_heads
        self.num_patches = self.seq_len // self.patch_size
        self.output_dim = args.output_dim
        self.num_quantiles = args.num_quantiles

        # Sparsity regularization on CLS token (for controlled sparsity sweep)
        self.cls_l1_weight = getattr(args, 'cls_l1_weight', 0.0)
        self.cls_topk = getattr(args, 'cls_topk', 0)  # 0 = disabled
        
        # Activation
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            from nn.models import Sine
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        # Patchify and embed
        self.patch_embed = nn.Conv1d(
            self.in_channels, 
            self.dim, 
            kernel_size=self.patch_size, 
            stride=self.patch_size
        )
        
        # Positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=self.heads,
            dim_feedforward=self.dim * 4,
            dropout=transformer_args.dropout_p,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.depth)
        
        # Decoder for reconstruction
        self.decoder = nn.ModuleList([
            nn.Linear(self.dim, self.dim * 2),
            nn.LayerNorm(self.dim * 2),
            self.activation,
            nn.Linear(self.dim * 2, self.patch_size * self.in_channels)
        ])
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim // 2),
            nn.BatchNorm1d(self.dim // 2),
            self.activation,
            nn.Dropout(transformer_args.dropout_p),
            nn.Linear(self.dim // 2, self.output_dim * self.num_quantiles)
        )
        
    def forward(self, x, y=None, meta=None,
                return_tokens: bool = False,
                return_all: bool = False):
        B = x.shape[0]
        
        # Handle input shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1, -2)
            
        # Patchify: [B, C, L] -> [B, num_patches, dim]
        x = self.patch_embed(x.float())  # [B, dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, dim]
        
        # Add cls token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, dim]
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Transformer encoding
        x_enc = self.transformer(x)
        
        # Split cls token and patches
        cls_output = x_enc[:, 0]  # [B, dim]
        patch_output = x_enc[:, 1:]  # [B, num_patches, dim]

        # Top-K sparsification on CLS token
        if self.cls_topk > 0 and self.cls_topk < self.dim:
            topk_vals, topk_idx = torch.topk(cls_output.abs(), self.cls_topk, dim=1)
            mask = torch.zeros_like(cls_output).scatter_(1, topk_idx, 1.0)
            cls_output = cls_output * mask

        # Regression from cls token
        output_reg = self.regressor(cls_output)
        
        # Reconstruction from patches
        x_dec = patch_output
        for layer in self.decoder:
            if isinstance(layer, nn.Linear):
                x_dec = layer(x_dec)
            else:
                x_dec = layer(x_dec)
        
        # Reshape to original size
        # Reshape to generated size (might be slightly smaller than seq_len due to floor division)
        x_dec = x_dec.reshape(B, self.num_patches, self.patch_size, self.in_channels)
        x_dec = x_dec.permute(0, 3, 1, 2)  # [B, C, num_patches, patch_size]
        output_dec = x_dec.reshape(B, self.in_channels, self.num_patches * self.patch_size)
        
        # Ensure correct length
        if output_dec.shape[-1] != self.seq_len:
            import torch.nn.functional as F
            output_dec = F.interpolate(output_dec, size=self.seq_len, mode='linear', align_corners=False)
        
        if self.in_channels == 1:
            output_dec = output_dec.squeeze(1)
        
        if return_all:
            # convenient dict API for experiments
            return {
                "regression": output_reg,
                "reconstruction": output_dec,
                "cls": cls_output,
                "tokens": x_enc,          # [B, 1+num_patches, dim]
                "patch_tokens": patch_output,
            }

        if return_tokens:
            # for pure encoder usage in probes
            return x_enc, cls_output, patch_output
        
        return output_reg, output_dec, cls_output


# ==========================
# Option 2: Hierarchical 2D Transformer
# ==========================

# class DualAxisTransformer(nn.Module):
#     """Process both channel and sequence dimensions explicitly"""
    
#     def __init__(self, args, conformer_args):
#         super().__init__()
        
#         # Parameters
#         self.in_channels = args.in_channels
#         self.channels = args.encoder_dims[-1] if hasattr(args, 'encoder_dims') else 256
#         self.seq_len = args.seq_len if hasattr(args, 'seq_len') else 1024
#         self.d_model = conformer_args.encoder_dim
#         self.num_heads = conformer_args.num_heads
#         self.output_dim = args.output_dim
#         self.num_quantiles = args.num_quantiles
#         self.stride = args.stride if hasattr(args, 'stride') else 4
        
#         # Activation
#         if args.activation == 'silu':
#             self.activation = nn.SiLU()
#         else:
#             self.activation = nn.ReLU()
        
#         # Initial CNN feature extraction
#         self.feature_extract = nn.Sequential(
#             nn.Conv1d(self.in_channels, self.channels // 4, 7, padding=3),
#             nn.BatchNorm1d(self.channels // 4),
#             self.activation,
#             nn.Conv1d(self.channels // 4, self.channels // 2, 5, padding=2),
#             nn.BatchNorm1d(self.channels // 2),
#             self.activation,
#             nn.Conv1d(self.channels // 2, self.channels, 3, padding=1),
#             nn.BatchNorm1d(self.channels),
#             self.activation
#         )
        
#         # Downsampling
#         self.downsample = nn.MaxPool1d(self.stride)
#         self.reduced_len = self.seq_len // self.stride
        
#         # Sequence-wise attention layers
#         self.seq_attention_layers = nn.ModuleList([
#             nn.MultiheadAttention(self.channels, self.num_heads, 
#                                  dropout=conformer_args.dropout_p, batch_first=True)
#             for _ in range(3)
#         ])
#         self.seq_norms = nn.ModuleList([nn.LayerNorm(self.channels) for _ in range(3)])
        
#         # Channel-wise attention layers
#         self.channel_attention_layers = nn.ModuleList([
#             nn.MultiheadAttention(self.reduced_len, self.num_heads // 2, 
#                                  dropout=conformer_args.dropout_p, batch_first=True)
#             for _ in range(3)
#         ])
#         self.channel_norms = nn.ModuleList([nn.LayerNorm(self.reduced_len) for _ in range(3)])
        
#         # Cross-dimensional fusion
#         self.fusion = nn.Sequential(
#             nn.LayerNorm(self.channels * self.reduced_len),
#             nn.Linear(self.channels * self.reduced_len, self.d_model),
#             self.activation,
#             nn.Dropout(conformer_args.dropout_p)
#         )
        
#         # ============= FIXED DECODER =============
#         # Option 1: Direct reconstruction from combined features
#         self.decoder_projection = nn.Linear(self.d_model, self.channels * self.reduced_len)
        
#         # Decoder with proper upsampling
#         self.decoder = nn.ModuleList()
        
#         # First layer: reshape and initial processing
#         self.decoder.append(nn.Sequential(
#             nn.Conv1d(self.channels, self.channels, 3, padding=1),
#             nn.BatchNorm1d(self.channels),
#             self.activation
#         ))
        
#         # Upsampling layers to restore original resolution
#         if self.stride > 1:
#             self.decoder.append(nn.Sequential(
#                 nn.ConvTranspose1d(self.channels, self.channels // 2, 
#                                    kernel_size=self.stride * 2, 
#                                    stride=self.stride, 
#                                    padding=self.stride // 2),
#                 nn.BatchNorm1d(self.channels // 2),
#                 self.activation
#             ))
            
#             self.decoder.append(nn.Sequential(
#                 nn.Conv1d(self.channels // 2, self.channels // 4, 3, padding=1),
#                 nn.BatchNorm1d(self.channels // 4),
#                 self.activation
#             ))
        
#         # Final projection to original channels
#         self.decoder.append(
#             nn.Conv1d(self.channels // 4 if self.stride > 1 else self.channels, 
#                       self.in_channels, 
#                       kernel_size=1)
#         )
        
#         # Regression head
#         self.regressor = nn.Sequential(
#             nn.Linear(self.d_model, self.d_model // 2),
#             nn.BatchNorm1d(self.d_model // 2),
#             self.activation,
#             nn.Dropout(conformer_args.dropout_p),
#             nn.Linear(self.d_model // 2, self.output_dim * self.num_quantiles)
#         )
        
#     def forward(self, x, y=None):
#         B = x.shape[0]
#         original_shape = x.shape
        
#         # Handle input shape
#         if len(x.shape) == 2:
#             x = x.unsqueeze(1)
#         elif len(x.shape) == 3 and x.shape[-1] == 1:
#             x = x.transpose(-1, -2)
        
#         # Extract features
#         x = self.feature_extract(x.float())  # [B, C, L]
#         x = self.downsample(x)  # [B, C, L//stride]
        
#         # Store for potential skip connections
#         x_enc_full = x.clone()
        
#         # Process with dual attention
#         # Sequence attention path
#         x_seq = x.transpose(1, 2)  # [B, L//stride, C]
#         for attn, norm in zip(self.seq_attention_layers, self.seq_norms):
#             x_seq_res = x_seq
#             x_seq = norm(x_seq)
#             x_seq_attn, _ = attn(x_seq, x_seq, x_seq)
#             x_seq = x_seq_res + x_seq_attn
        
#         # Channel attention path
#         x_chan = x  # [B, C, L//stride]
#         for attn, norm in zip(self.channel_attention_layers, self.channel_norms):
#             x_chan_res = x_chan
#             x_chan = norm(x_chan)
#             x_chan_attn, _ = attn(x_chan, x_chan, x_chan)
#             x_chan = x_chan_res + x_chan_attn
        
#         # Combine both views
#         x_combined = x_seq.transpose(1, 2) + x_chan  # [B, C, L//stride]
        
#         # Flatten and fuse for bottleneck
#         x_flat = x_combined.reshape(B, -1)  # [B, C * L//stride]
#         x_enc = self.fusion(x_flat)  # [B, d_model]
        
#         # Regression
#         output_reg = self.regressor(x_enc)
        
#         # ============= FIXED RECONSTRUCTION =============
#         # Project back to spatial dimensions
#         x_dec = self.decoder_projection(x_enc)  # [B, C * L//stride]
#         x_dec = x_dec.reshape(B, self.channels, self.reduced_len)  # [B, C, L//stride]
        
#         # Add residual from encoder if desired
#         x_dec = x_dec + x_enc_full * 0.5  # Weighted residual connection
        
#         # Apply decoder layers
#         for layer in self.decoder:
#             x_dec = layer(x_dec)
        
#         # Ensure correct output shape
#         if x_dec.shape[-1] != self.seq_len:
#             x_dec = F.interpolate(x_dec, size=self.seq_len, mode='linear', align_corners=False)
        
#         # Match original input shape
#         if self.in_channels == 1:
#             output_dec = x_dec.squeeze(1)
#         else:
#             output_dec = x_dec
            
#         # Ensure output shape matches input shape for reconstruction loss
#         if len(original_shape) == 2:
#             if len(output_dec.shape) == 3:
#                 output_dec = output_dec.squeeze(1)
        
#         return output_reg, output_dec, x_enc


# Alternative: More sophisticated decoder with multiple upsampling stages
class DualAxisTransformer(nn.Module):
    """Version with progressive upsampling decoder"""
    
    def __init__(self, args, transformer_args):
        super().__init__()
        
        # Parameters (same as before)
        self.in_channels = args.in_channels
        self.channels = args.encoder_dims[-1] if hasattr(args, 'encoder_dims') else 256
        self.seq_len = args.seq_len if hasattr(args, 'seq_len') else 1024
        self.d_model = transformer_args.encoder_dim
        self.num_heads = transformer_args.num_heads
        self.output_dim = args.output_dim
        self.num_quantiles = args.num_quantiles
        self.stride = args.stride if hasattr(args, 'stride') else 4
        
        # Activation
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Feature extraction and attention layers (same as before)
        self.feature_extract = nn.Sequential(
            nn.Conv1d(self.in_channels, self.channels // 4, 7, padding=3),
            nn.BatchNorm1d(self.channels // 4),
            self.activation,
            nn.Conv1d(self.channels // 4, self.channels // 2, 5, padding=2),
            nn.BatchNorm1d(self.channels // 2),
            self.activation,
            nn.Conv1d(self.channels // 2, self.channels, 3, padding=1),
            nn.BatchNorm1d(self.channels),
            self.activation
        )
        
        self.downsample = nn.MaxPool1d(self.stride)
        self.reduced_len = self.seq_len // self.stride
        
        # Attention layers (same structure)
        self.seq_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(self.channels, self.num_heads, 
                                 dropout=transformer_args.dropout_p, batch_first=True)
            for _ in range(3)
        ])
        self.seq_norms = nn.ModuleList([nn.LayerNorm(self.channels) for _ in range(3)])
        
        self.channel_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(self.reduced_len, self.num_heads // 2, 
                                 dropout=transformer_args.dropout_p, batch_first=True)
            for _ in range(3)
        ])
        self.channel_norms = nn.ModuleList([nn.LayerNorm(self.reduced_len) for _ in range(3)])
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.channels * self.reduced_len),
            nn.Linear(self.channels * self.reduced_len, self.d_model),
            self.activation,
            nn.Dropout(transformer_args.dropout_p)
        )
        
        # ============= PROGRESSIVE DECODER =============
        # Calculate number of upsampling stages needed
        num_upsample_stages = int(math.log2(self.stride)) if self.stride > 1 else 0
        
        # Initial projection from bottleneck
        self.decoder_init = nn.Sequential(
            nn.Linear(self.d_model, self.channels * (self.reduced_len // 4)),
            self.activation
        )
        
        # Progressive upsampling
        self.decoder_stages = nn.ModuleList()
        current_channels = self.channels
        current_len = self.reduced_len // 4
        
        # Add initial conv to process projected features
        self.decoder_stages.append(nn.Sequential(
            nn.Conv1d(current_channels, current_channels, 3, padding=1),
            nn.BatchNorm1d(current_channels),
            self.activation
        ))
        
        # Progressive upsampling to original length
        for i in range(num_upsample_stages + 2):  # +2 to account for initial downsize
            next_channels = max(current_channels // 2, self.in_channels)
            self.decoder_stages.append(nn.Sequential(
                nn.ConvTranspose1d(current_channels, next_channels,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm1d(next_channels),
                self.activation if i < num_upsample_stages + 1 else nn.Identity()
            ))
            current_channels = next_channels
            current_len *= 2
        
        # Final projection
        self.decoder_final = nn.Conv1d(current_channels, self.in_channels, 1)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.BatchNorm1d(self.d_model // 2),
            self.activation,
            nn.Dropout(conformer_args.dropout_p),
            nn.Linear(self.d_model // 2, self.output_dim * self.num_quantiles)
        )
        
    def forward(self, x, y=None):
        B = x.shape[0]
        original_shape = x.shape
        
        # Handle input shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1, -2)
        
        # Feature extraction and attention (same as before)
        x = self.feature_extract(x.float())
        x = self.downsample(x)
        
        # Dual attention processing
        x_seq = x.transpose(1, 2)
        for attn, norm in zip(self.seq_attention_layers, self.seq_norms):
            x_seq_res = x_seq
            x_seq = norm(x_seq)
            x_seq_attn, _ = attn(x_seq, x_seq, x_seq)
            x_seq = x_seq_res + x_seq_attn
        
        x_chan = x
        for attn, norm in zip(self.channel_attention_layers, self.channel_norms):
            x_chan_res = x_chan
            x_chan = norm(x_chan)
            x_chan_attn, _ = attn(x_chan, x_chan, x_chan)
            x_chan = x_chan_res + x_chan_attn
        
        x_combined = x_seq.transpose(1, 2) + x_chan
        x_flat = x_combined.reshape(B, -1)
        x_enc = self.fusion(x_flat)
        
        # Regression
        output_reg = self.regressor(x_enc)
        
        # Progressive decoder
        x_dec = self.decoder_init(x_enc)
        x_dec = x_dec.reshape(B, self.channels, -1)
        
        for stage in self.decoder_stages:
            x_dec = stage(x_dec)
        
        x_dec = self.decoder_final(x_dec)
        
        # Ensure correct length
        if x_dec.shape[-1] != self.seq_len:
            x_dec = F.interpolate(x_dec, size=self.seq_len, mode='linear', align_corners=False)
        
        # Format output
        if self.in_channels == 1:
            output_dec = x_dec.squeeze(1)
        else:
            output_dec = x_dec
        
        return output_reg, output_dec, x_enc


# ==========================
# Option 3: U-Net with Axial Attention
# ==========================

class AxialAttentionBlock(nn.Module):
    """Attention block that can work along specified axis"""
    
    def __init__(self, channels, num_heads=4, axis='sequence'):
        super().__init__()
        self.axis = axis
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, x):
        # x: [B, C, L]
        if self.axis == 'sequence':
            x_t = x.transpose(1, 2)  # [B, L, C]
            x_norm = self.norm(x_t)
            x_attn, _ = self.attention(x_norm, x_norm, x_norm)
            x_t = x_t + x_attn
            return x_t.transpose(1, 2)
        else:  # channel axis
            x_norm = self.norm(x)
            x_attn, _ = self.attention(x_norm, x_norm, x_norm)
            return x + x_attn


class SpectralUNetAxial(nn.Module):
    """U-Net with axial attention for better reconstruction"""
    
    def __init__(self, args, conformer_args):
        super().__init__()
        
        # Parameters
        self.in_channels = args.in_channels
        self.channels = args.encoder_dims if hasattr(args, 'encoder_dims') else [32, 64, 128, 256]
        self.d_model = conformer_args.encoder_dim
        self.output_dim = args.output_dim
        self.num_quantiles = args.num_quantiles
        
        # Activation
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        # Encoder path with axial attention
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        for i, (in_c, out_c) in enumerate(zip([self.in_channels] + self.channels[:-1], self.channels)):
            encoder_block = nn.ModuleList([
                nn.Conv1d(in_c, out_c, 3, padding=1),
                nn.BatchNorm1d(out_c),
                self.activation,
                nn.Conv1d(out_c, out_c, 3, padding=1),
                nn.BatchNorm1d(out_c),
                self.activation
            ])
            
            # Add axial attention for deeper layers
            if i > 1:
                encoder_block.append(AxialAttentionBlock(out_c, num_heads=4, axis='sequence'))
                
            self.encoders.append(encoder_block)
            self.pools.append(nn.MaxPool1d(2))
        
        # Bottleneck with full transformer
        bottleneck_layer = nn.TransformerEncoderLayer(
            d_model=self.channels[-1],
            nhead=conformer_args.num_heads,
            dim_feedforward=self.channels[-1] * 4,
            dropout=conformer_args.dropout_p,
            batch_first=True
        )
        self.bottleneck = nn.TransformerEncoder(bottleneck_layer, num_layers=2)
        
        # Global pooling for embedding
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Linear(self.channels[-1], self.d_model)
        
        # Decoder path with skip connections
        self.decoders = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        reversed_channels = self.channels[::-1]
        for i, (in_c, out_c) in enumerate(zip(reversed_channels[:-1], reversed_channels[1:])):
            self.upsamples.append(
                nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2)
            )
            
            decoder_block = nn.ModuleList([
                nn.Conv1d(in_c, out_c, 3, padding=1),  # in_c for concatenated skip
                nn.BatchNorm1d(out_c),
                self.activation,
                nn.Conv1d(out_c, out_c, 3, padding=1),
                nn.BatchNorm1d(out_c),
                self.activation
            ])
            
            # Add axial attention for earlier decoder layers
            if i < 2:
                decoder_block.append(AxialAttentionBlock(out_c, num_heads=4, axis='sequence'))
                
            self.decoders.append(decoder_block)
        
        # Final reconstruction layer
        self.final_conv = nn.Conv1d(self.channels[0], self.in_channels, kernel_size=1)
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.BatchNorm1d(self.d_model // 2),
            self.activation,
            nn.Dropout(conformer_args.dropout_p),
            nn.Linear(self.d_model // 2, self.output_dim * self.num_quantiles)
        )
        
    def forward(self, x, y=None):
        B = x.shape[0]
        
        # Handle input shape
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1, -2)
        
        x = x.float()
        
        # Encoder path - store skip connections
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            for layer in encoder:
                if isinstance(layer, AxialAttentionBlock):
                    x = layer(x)
                elif isinstance(layer, nn.Module):
                    x = layer(x)
            skip_connections.append(x)
            x = pool(x)
        
        # Bottleneck
        x_seq = x.transpose(1, 2)  # [B, L, C]
        x_seq = self.bottleneck(x_seq)
        x = x_seq.transpose(1, 2)  # [B, C, L]
        
        # Global feature for regression
        x_global = self.global_pool(x).squeeze(-1)  # [B, C]
        x_enc = self.projection(x_global)  # [B, d_model]
        output_reg = self.regressor(x_enc)
        
        # Decoder path with skip connections
        for i, (decoder, upsample) in enumerate(zip(self.decoders, self.upsamples)):
            x = upsample(x)
            
            # Get skip connection and concatenate
            skip = skip_connections[-(i + 2)]  # Get corresponding encoder output
            
            # Ensure sizes match (might need to crop/pad)
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                if diff > 0:
                    x = F.pad(x, (0, diff))
                else:
                    skip = F.pad(skip, (0, -diff))
                    
            x = torch.cat([x, skip], dim=1)
            
            for layer in decoder:
                if isinstance(layer, AxialAttentionBlock):
                    x = layer(x)
                else:
                    x = layer(x)
        
        # Final reconstruction
        output_dec = self.final_conv(x)
        
        if self.in_channels == 1:
            output_dec = output_dec.squeeze(1)
        
        return output_reg, output_dec, x_enc
