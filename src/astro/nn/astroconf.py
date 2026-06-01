from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.functional import softmax
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from .Modules.conformer import ConformerEncoder, ConformerDecoder
from .Modules.mhsa_pro import RotaryEmbedding, ContinuousRotaryEmbedding
from .Modules.cnn import ResNetBlock
# from .Modules.NCE import Net
from .Modules.ResNet18 import ResNet18

class Astroconformer(nn.Module):
  def __init__(self, args) -> None:
    super(Astroconformer, self).__init__()
    self.head_size = args.encoder_dim // args.num_heads
    self.rotary_ndims = int(self.head_size * 0.5)
    print("extractor stride: ", args.stride)
    self.extractor = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
            kernel_size = args.stride, out_channels = args.encoder_dim, stride = args.stride, padding = 0, bias = True),
                    nn.BatchNorm1d(args.encoder_dim),
                    nn.SiLU(),
    )
    
    self.use_continuous_rope = getattr(args, 'use_continuous_rope', False)
    if self.use_continuous_rope:
        print("Using Continuous RoPE in Astroconformer")
        sequence_scale = getattr(args, 'sequence_scale', 1.0)
        self.pe = ContinuousRotaryEmbedding(self.rotary_ndims, sequence_scale)
    else:
        self.pe = RotaryEmbedding(self.rotary_ndims)
    
    # [NEW] Optional Time Embedding
    self.use_time_embedding = getattr(args, 'use_time_embedding', False)
    if self.use_time_embedding:
        print("Using Time Embedding in Astroconformer")
        # Continuous time embedding via MLP
        self.time_embedding = nn.Sequential(
            nn.Linear(1, args.encoder_dim // 2),
            nn.SiLU(),
            nn.Linear(args.encoder_dim // 2, args.encoder_dim)
        )
    
    self.encoder = ConformerEncoder(args)
    self.output_dim = args.encoder_dim
    self.avg_output = args.avg_output
    
    if not args.encoder_only:
        self.pred_layer = nn.Sequential(
            nn.Linear(args.encoder_dim, args.encoder_dim),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(args.encoder_dim,args.output_dim),
        )
    else:
        self.pred_layer = nn.Identity()
    if getattr(args, 'mean_label', False):
      self.pred_layer[3].bias.data.fill_(args.mean_label)

    self.init_weights()

  def init_weights(self):
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
    
  def forward(self, inputs: Tensor, time: Tensor = None) -> Tensor:
    x = inputs #initial input_size: [B, L]
    if len(x.shape) == 2:
        x = x.unsqueeze(1) # x: [B, 1, L]
    x = self.extractor(x) # x: [B, encoder_dim, L]
    x = x.permute(0,2,1) # x: [B, L, encoder_dim]
    
    # [NEW] Handle time resolution mismatch (extractor downsamples flux but not time)
    if time is not None:
        seq_len_out = x.shape[1]
        seq_len_in = time.shape[1]
        if seq_len_in != seq_len_out:
            # Assume uniform downsampling factor (e.g., from extractor stride)
            scale = seq_len_in // seq_len_out
            if scale > 1:
                # Use mean pooling to get representative time for each patch
                # time: [B, L_in] -> [B, 1, L_in]
                time = torch.nn.functional.avg_pool1d(
                    time.unsqueeze(1), 
                    kernel_size=scale, 
                    stride=scale
                ).squeeze(1)
                # Ensure length matches exactly (handle rounding)
                if time.shape[1] > seq_len_out:
                    time = time[:, :seq_len_out]
                elif time.shape[1] < seq_len_out:
                     # Pad with last value if slightly short
                    time = torch.nn.functional.pad(time, (0, seq_len_out - time.shape[1]), mode='replicate')

    # [NEW] Add Time Embedding
    if self.use_time_embedding and time is not None:
        # time: [B, L] -> [B, L, 1]
        t_emb = self.time_embedding(time.unsqueeze(-1)) # [B, L, encoder_dim]
        # Add to x
        x = x + t_emb

    if self.use_continuous_rope:
        if time is not None:
            RoPE = self.pe(time)
        else:
            # Fallback for missing time: assume uniform steps [0, 1, ..., L-1]
            seq_len = x.shape[1]
            uniform_time = torch.arange(seq_len, device=x.device).float().unsqueeze(0).expand(x.shape[0], -1)
            RoPE = self.pe(uniform_time)
    else:
        RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
    x = self.encoder(x, RoPE) # x: [B, L, encoder_dim]
    # memory = x.clone()
    # print("x shape: ", x.shape)
    if self.avg_output:
      x = x.mean(dim=1) # x: [B, encoder_dim]
      # print("x shape: ", x.shape)
      # attn = softmax(self.attnpool(x), dim=1) # attn: [B, L, 1]
      # x = torch.matmul(x.permute(0,2,1), attn).squeeze(-1) # x: [B, encoder_dim]
      x = self.pred_layer(x) # x: [B, output_dim]
    return x

class AstroDecoder(nn.Module):
  def __init__(self, args) -> None:
    super(AstroDecoder, self).__init__()
    self.embedding = nn.Linear(args.in_channels, args.decoder_dim)
    self.head_size = args.decoder_dim // args.num_heads
    self.rotary_ndims = int(self.head_size * 0.5)    
    self.pe = RotaryEmbedding(self.rotary_ndims)
    self.decoder = ConformerDecoder(args)
    
  def forward(self, tgt: Tensor, memory: Tensor) -> Tensor:
    if len(tgt.shape)==2:
      tgt = tgt.unsqueeze(-1)
    x = self.embedding(tgt) #initial input_size: [B, L, encoder_dim]

    RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
    x = self.decoder(x, memory, RoPE) # x: [B, L, encoder_dim]
    return x

class AstroEncoderDecoder(nn.Module):
  def __init__(self, args) -> None:
    super(AstroEncoderDecoder, self).__init__()
    self.encoder = Astroconformer(args)
    self.encoder.pred_layer = nn.Identity()
    self.decoder = AstroDecoder(args)
    
  def forward(self, inputs: Tensor, tgt: Tensor) -> Tensor:
    x, memory = self.encoder(inputs)
    x = self.decoder(tgt, memory)
    return x.mean(dim=-1)

class ResNetBaseline(nn.Module):
  def __init__(self, args) -> None:
    super().__init__()
    
    self.embedding = nn.Sequential(nn.Conv1d(in_channels = args.in_channels,
            kernel_size=3, out_channels = args.encoder_dim, stride=1, padding = 'same', bias = False),
                    nn.BatchNorm1d(args.encoder_dim),
                    nn.SiLU(),
    )
    
    self.layers = nn.ModuleList([ResNetBlock(args)
      for _ in range(args.num_layers)])
    
    self.pred_layer = nn.Sequential(
        nn.Linear(args.encoder_dim, args.encoder_dim),
        nn.SiLU(),
        nn.Dropout(p=0.3),
        nn.Linear(args.encoder_dim,1),
    )
    if getattr(args, 'mean_label', False):
      self.pred_layer[3].bias.fill_(args.mean_label)
    
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x.unsqueeze(1)
    x = self.embedding(x)
    for m in self.layers:
      x = m(x)
    x = x.mean(dim=-1)
    x = self.pred_layer(x)
    return x
    
model_dict = {
          'Astroconformer': Astroconformer,
          'ResNetBaseline': ResNetBaseline,
          'ResNet18': ResNet18,
      }
