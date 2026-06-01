from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.Modules.conformer import ConformerEncoder, ConformerDecoder
from nn.Modules.mhsa_pro import RotaryEmbedding, ContinuousRotaryEmbedding
from nn.Modules.flash_mhsa import MHA as Flash_Mha
from nn.Modules.mlp import Mlp as MLP
from nn.simsiam import projection_MLP, SimSiam

import numbers
import torch.nn.init as init
from typing import Union, List, Optional, Tuple
from torch import Size, Tensor



def get_activation(args):
    
    if args.activation == 'silu':
        return nn.SiLU()
    elif args.activation == 'sine':
        return Sine(w0=args.sine_w0)
    elif args.activation == 'relu':
        return nn.ReLU()
    elif args.activation == 'gelu':
        return nn.GELU()
    else:
        return nn.ReLU()

class MLPEncoder(nn.Module):
    def __init__(self, args):
        """
        Initialize an MLP with hidden layers, BatchNorm, and Dropout.

        Args:
            input_dim (int): Dimension of the input features.
            hidden_dims (list of int): List of dimensions for hidden layers.
            output_dim (int): Dimension of the output.
            dropout (float): Dropout probability (default: 0.0).
        """
        super(MLPEncoder, self).__init__()
        
        layers = []
        prev_dim = args.input_dim
        
        # Add hidden layers
        for hidden_dim in args.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim)) 
            layers.append(nn.SiLU())  
            if args.dropout > 0.0:
                layers.append(nn.Dropout(args.dropout))  
            prev_dim = hidden_dim
        self.model = nn.Sequential(*layers)
        self.output_dim = hidden_dim
        
    
    def forward(self, x, y):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        x = self.model(x)
        x = x.mean(-1)
        return x

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], eps: float = 1e-5, bias: bool = False) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        var = input.pow(2).mean(dim=-1, keepdim=True) + self.eps
        input_norm = input * torch.rsqrt(var)

        rmsnorm = self.weight * input_norm
        
        if self.bias is not None:
            rmsnorm = rmsnorm + self.bias

        return rmsnorm

class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)

class ConvBlock(nn.Module):
  def __init__(self, args, num_layer) -> None:
    super().__init__()
    self.activation = get_activation(args)
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
    if in_channels != out_channels:
        self.skip_connection = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        )
    else:
        self.skip_connection = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.layers(x)
    if self.skip_connection is not None:
        out = out + self.skip_connection(x)
    return out

class CNNEncoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        print("Using CNN encoder wit activation: ", args.activation, 'args avg_output: ', args.avg_output)
        self.activation = get_activation(args)
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
            x = m(x)
            if x.shape[-1] > self.min_seq_len:
                x = self.pool(x)
        if self.avg_output:
            x = x.mean(dim=-1)
        else:
            x = x.permute(0,2,1)
        return x


class LSTMEncoder(nn.Module):
    def __init__(self, args):
        super(LSTMEncoder, self).__init__()
        self.t_features = args.seq_len//args.stride
        self.activation = get_activation(args)
        # self.conv = Conv2dSubampling(in_channels=in_channels, out_channels=self.encoder_dims[0])
        self.conv1 = nn.Conv1d(in_channels=args.in_channels, out_channels=args.encoder_dims[0], kernel_size=args.kernel_size, padding='same', stride=1)
        self.pool = nn.MaxPool1d(kernel_size=args.stride)
        self.skip = nn.Conv1d(in_channels=args.in_channels, out_channels=args.encoder_dims[0], kernel_size=1, padding=0, stride=args.stride)
        self.drop = nn.Dropout1d(p=args.dropout_p)
        self.batchnorm1 = nn.BatchNorm1d(args.encoder_dims[0])
                    
        self.lstm = nn.LSTM(args.encoder_dims[0], args.encoder_dims[1], num_layers=args.num_layers,
                            batch_first=True, bidirectional=True, dropout=args.dropout_p)
       
        self.output_dim = args.encoder_dims[1]*2
        self.in_channels = args.in_channels
            #     torch.set_rng_state(rng_state)

    def lstm_attention(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        # Here we assume q_dim == k_dim (dot product attention)
        scale = 1/(keys.size(-1) ** -0.5)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return linear_combination

    def forward(self, x, return_cell=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1,-2)
        x = x.float()
        skip = self.skip(x)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(x)))))
        x = x + skip[:, :, :x.shape[-1]] # [B, C, L//stride]
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2) # [B, L//stride, C]
        x_f,(h_f,c_f) = self.lstm(x) # [B, L//stride, 2*hidden_size], [2*num_layers, B, hidden_size], [2*num_layers, B, hidden_size
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1) # [B, 2*hidden_szie]
        features = self.lstm_attention(c_f, x_f, x_f) # [B, 2*hidden_size]
        if return_cell:
            return features, h_f, c_f
        return features

      
class CNNDecoder(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        print("Using CNN decoder with activation: ", args.activation)
        
        # Reverse the encoder dimensions for upsampling
        decoder_dims = args.encoder_dims[::-1]
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        # Initial embedding layer to expand the compressed representation
        self.initial_expand = nn.Linear(decoder_dims[0], decoder_dims[0] * 4)
        
        # Transposed Convolutional layers for upsampling
        self.layers = nn.ModuleList()
        for i in range(args.num_layers):
            if i  < len(decoder_dims) - 1:
                in_channels = decoder_dims[i] 
                out_channels = decoder_dims[i+1]
            else:
                in_channels = decoder_dims[-1]
                out_channels = decoder_dims[-1]
            
            # Transposed Convolution layer
            layer = nn.Sequential(
                nn.ConvTranspose1d(in_channels=in_channels, 
                                   out_channels=out_channels, 
                                   kernel_size=4, 
                                   stride=2, 
                                   padding=1, 
                                   bias=False),
                nn.BatchNorm1d(out_channels),
                self.activation
            )
            self.layers.append(layer)
        
        # Final layer to match original input channels
        self.final_conv = nn.ConvTranspose1d(in_channels=decoder_dims[-1], 
                                             out_channels=1, 
                                             kernel_size=3, 
                                             stride=1, 
                                             padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expand the compressed representation
        # x = self.initial_expand(x)
        # x = x.unsqueeze(-1)  # Add sequence dimension
        
        # Apply transposed convolution layers
        x = x.float()
        for layer in self.layers:
            x = layer(x)
        
        # Final convolution to get back to original input channels
        x = self.final_conv(x)
        
        return x.squeeze()

class CNNEncoderDecoder(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        
        self.encoder = MultiEncoder(args, conformer_args)
        self.decoder = CNNDecoder(args)
            
    def forward(self, x, y=None):
        # Encode the input
        encoded, _ = self.encoder(x)
        # if self.transformer is not None:
        # Decode the compressed representation
        reconstructed = self.decoder(encoded)
        
        return reconstructed


class CNNRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        self.encoder = MultiEncoder(args, conformer_args)
        if args.freeze_encoder:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False
         
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()
        
        self.regressor = nn.Sequential(
            nn.Linear(conformer_args.encoder_dim, conformer_args.encoder_dim//2),
            nn.BatchNorm1d(conformer_args.encoder_dim//2),
            self.activation,
            nn.Dropout(0.2),

            nn.Linear(conformer_args.encoder_dim//2, conformer_args.encoder_dim//4),
            nn.BatchNorm1d(conformer_args.encoder_dim//4),
            self.activation,
            nn.Dropout(0.2),

            nn.Linear(conformer_args.encoder_dim//4, conformer_args.encoder_dim//8),
            nn.BatchNorm1d(conformer_args.encoder_dim//8),
            self.activation,
            nn.Dropout(0.2),

            nn.Linear(conformer_args.encoder_dim//8, args.output_dim*args.num_quantiles)
        )
    
    def forward(self, x):
        # Encode the input
        # x = self.backbone(x)
        # x = x.unsqueeze(1)
        # # x = x.permute(0,2,1)
        # RoPE = self.pe(x, x.shape[1]) # RoPE: [2, B, L, encoder_dim], 2: sin, cos
        # x = self.encoder(x, RoPE).squeeze(1)

        # if self.return_logits:
        #     return x

        # x = self.transformer(x)
        # x = x.sum(dim=1)
        x, _ = self.encoder(x)
        output = self.regressor(x)
        
        return output


class MultiTaskRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        
        super().__init__()
        self.encoder = MultiEncoder(args, transformer_args)
        self.decoder = CNNDecoder(args)
        # self.projector = projection_MLP(conformer_args.encoder_dim)
        
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()

        self.avg_output = args.avg_output
        encoder_dim = transformer_args.encoder_dim
        self.output_dim = encoder_dim
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim//2),
            nn.BatchNorm1d(encoder_dim//2),
            self.activation,
            nn.Dropout(transformer_args.dropout_p),
            nn.Linear(encoder_dim//2, args.output_dim*args.num_quantiles)
        )
    
    def forward(self, x, y=None):
        x_enc, x = self.encoder(x)
        x = x.permute(0,2,1)
        output_reg = self.regressor(x_enc)

        output_dec = self.decoder(x)
        return output_reg, output_dec, x_enc

class DualEncoder(nn.Module):
    def __init__(self, args, conformer_args):
        
        super().__init__()
        self.encoder1 = MultiEncoder(args, conformer_args)
        self.encoder2 = MultiEncoder(args, conformer_args)
                
        if args.activation == 'silu':
            self.activation = nn.SiLU()
        elif args.activation == 'sine':
            self.activation = Sine(w0=args.sine_w0)
        else:
            self.activation = nn.ReLU()

        self.avg_output = args.avg_output
        self.output_dim = conformer_args.encoder_dim
        
    
    def forward(self, x1, x2):
        x_enc1, x1 = self.encoder1(x1)
        x1 = x1.permute(0,2,1)
        x_enc2, x2 = self.encoder2(x2)
        x2 = x2.permute(0,2,1)
        return {
                'x_enc1': x_enc1,
                'x_enc2': x_enc2,
                'x1': x1,
                'x2': x2
                }

class CausalModel(nn.Module):
    def __init__(self, physics_model, noise_model, decoder,
                physics_out_dim, num_quantiles, noise_out_dim):
        super().__init__()
        self.physics_model = physics_model
        self.noise_model = noise_model
        self.decoder = decoder
        physics_dim = physics_model.output_dim
        noise_dim = noise_model.output_dim
        self.physics_regressor = nn.Sequential(
            nn.Linear(physics_dim, physics_dim//2),
            nn.BatchNorm1d(physics_dim//2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(physics_dim//2, physics_out_dim*num_quantiles)
        )
        self.noise_regressor = nn.Sequential(
            nn.Linear(noise_dim, noise_dim//2),
            nn.BatchNorm1d(noise_dim//2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(noise_dim//2, noise_out_dim)
        )

    def forward(self, anchor, positive, negative):
        out_physics = self.physics_model(anchor, positive)
        out_noise = self.noise_model(anchor, negative)
        physics_reg = self.physics_regressor(out_physics['x_enc1'] + out_physics['x_enc2'])
        noise_reg = self.noise_regressor(out_noise['x_enc1'] + out_noise['x_enc2'])
        dec = self.decoder(out_physics['x1'] + out_noise['x1'])
        return {'physics_reg': physics_reg, 'noise_reg': noise_reg,
                'dec': dec, 'out_physics': out_physics, 'out_noise': out_noise}

class SimpleRegressor(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        self.encoder = encoder
        encoder_dim = args.encoder_dim
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim//2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(encoder_dim//2, args.output_dim*args.num_quantiles)
        )
    def forward(self, x, x_dual=None, w_tune=None):
        x_enc = self.encoder(x)
        if isinstance(x_enc, tuple):
            x_enc = x_enc[0]
        out = self.regressor(x_enc)
        return out, x_enc



class ACFDecoder(nn.Module):
    def __init__(self, input_dim, acf_length, hidden_dims=[512, 256, 128]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, acf_length))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)

class MambaConformerEncoder(nn.Module):
    """
    Encoder combining Mamba for long-range dependencies with Conformer for local patterns
    """
    def __init__(self, args):
        super().__init__()
        
        # Input embedding
        self.input_dim = args.seq_len
        self.d_model = args.d_model
        self.in_channels = getattr(args, 'in_channels', 1)
        self.activation = nn.GELU()
        
        # Input projection
        self.input_proj = nn.Linear(self.in_channels, self.d_model)
        
        # Mamba layer for long-range dependencies
        force_mamba = getattr(args, 'force_mamba', False)
        try:
            from mamba_ssm import Mamba
            self.mamba = Mamba(
                d_model=self.d_model,
                d_state=args.d_state,
                d_conv=args.d_conv,
                expand=args.expand,
            )
            self.has_mamba = True
        except ImportError:
            if force_mamba:
                raise ImportError("Mamba required but mamba_ssm not found in this environment.")
            print("Mamba not available, using LSTM instead")
            self.mamba = nn.LSTM(
                input_size=self.d_model,
                hidden_size=self.d_model,
                num_layers=2,
                batch_first=True,
                bidirectional=False
            )
            self.has_mamba = False
        
        # Lightweight transformer layers with RoPE
        from nn.Modules.mhsa_pro import RotaryEmbedding
        
        # Rotary positional embedding
        head_size = self.d_model // args.num_heads
        rotary_ndims = int(head_size * 0.5)
        self.pe = RotaryEmbedding(rotary_ndims)
        
        # Transformer layers
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=args.num_heads,
            dim_feedforward=self.d_model * 4,
            dropout=args.dropout_p,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer, 
            num_layers=args.num_conformer_layers
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, args.output_dim),
            self.activation
        )
        
        self.output_dim = args.output_dim
        self.avg_output = getattr(args, 'avg_output', True)
        
    def forward(self, x):
        import time
        start_time = time.time()
        
        # x expected as [B, seq_len] or [B, c, seq_len]
        if len(x.shape) == 3:
            # Transform to [B, seq_len, c] for linear embedding
            x = x.permute(0, 2, 1)
        elif len(x.shape) == 2:
            x = x.unsqueeze(-1)  # [B, seq_len, 1]
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
            
        # Project to model dimension: [B, seq_len, C] -> [B, seq_len, d_model]
        proj_start = time.time()
        x = self.input_proj(x)
        proj_time = time.time() - proj_start
        
        # Mamba for long-range dependencies
        mamba_start = time.time()
        if self.has_mamba:
            x = self.mamba(x)  # [B, seq_len, d_model]
        else:
            x, _ = self.mamba(x)  # LSTM fallback
        mamba_time = time.time() - mamba_start
        
        # Add rotary positional embedding
        rope_start = time.time()
        RoPE = self.pe(x, x.shape[1]).nan_to_num(0)  # [2, B, L, d_model//2]
        rope_time = time.time() - rope_start
        
        # Apply transformer with RoPE (note: standard transformer doesn't use RoPE directly)
        # For simplicity, we'll just use the transformer without RoPE for now
        transformer_start = time.time()
        if len(self.transformer.layers) > 0:
            if self.training:
                # Use gradient checkpointing during training to save memory
                x = torch.utils.checkpoint.checkpoint(self.transformer, x, use_reentrant=False)
            else:
                x = self.transformer(x)  # [B, seq_len, d_model]
        transformer_time = time.time() - transformer_start
        
        # Global average pooling: [B, seq_len, d_model] -> [B, d_model] if avg_output is True
        pool_start = time.time()
        if self.avg_output:
            x = x.mean(dim=1)
        pool_time = time.time() - pool_start
        
        # Output projection
        output_start = time.time()
        x = self.output_proj(x)  # [B, output_dim] if pooled, [B, L, output_dim] if not
        output_time = time.time() - output_start
        
        total_time = time.time() - start_time
        
        # Print timing info every 100 forward passes
        if hasattr(self, '_forward_count'):
            self._forward_count += 1
        else:
            self._forward_count = 1
            
        # if self._forward_count % 100 == 0:
        #     print(f"[ENCODER TIMING] Total: {total_time:.4f}s")
        #     print(f"  - Projection: {proj_time:.4f}s ({proj_time/total_time*100:.1f}%)")
        #     print(f"  - LSTM/Mamba: {mamba_time:.4f}s ({mamba_time/total_time*100:.1f}%)")
        #     print(f"  - RoPE: {rope_time:.4f}s ({rope_time/total_time*100:.1f}%)")
        #     print(f"  - Transformer: {transformer_time:.4f}s ({transformer_time/total_time*100:.1f}%)")
        #     print(f"  - Pooling: {pool_time:.4f}s ({pool_time/total_time*100:.1f}%)")
        #     print(f"  - Output: {output_time:.4f}s ({output_time/total_time*100:.1f}%)")
        
        return x

class MambaConformerSimSiam(nn.Module):
    """
    SimSiam architecture using MambaConformer encoder with ACF reconstruction
    """
    def __init__(self, args):
        super().__init__()
        
        # Encoder
        encoder_args = type('Args', (), {
            'seq_len': args.seq_len,
            'd_model': args.d_model,
            'd_state': getattr(args, 'd_state', 16),
            'd_conv': getattr(args, 'd_conv', 4),
            'expand': getattr(args, 'expand', 2),
            'num_conformer_layers': getattr(args, 'num_conformer_layers', 4),
            'num_heads': getattr(args, 'num_heads', 8),
            'dropout_p': args.dropout_p,
            'output_dim': args.encoder_output_dim
        })()
        
        self.encoder = MambaConformerEncoder(encoder_args)
        
        # SimSiam components
        from nn.simsiam import SimSiam
        self.simsiam = SimSiam(self.encoder)
        
        # Task heads
        encoder_dim = self.simsiam.output_dim * 2  # Concatenated z1 + z2
        self.output_dim = encoder_dim
        
        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.BatchNorm1d(encoder_dim // 2),
            nn.GELU(),
            nn.Dropout(args.dropout_p),
            nn.Linear(encoder_dim // 2, args.output_dim * args.num_quantiles)
        )
        
        # Classification head
        if hasattr(args, 'num_cls') and args.num_cls > 0:
            self.classifier = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.BatchNorm1d(encoder_dim // 2),
                nn.GELU(),
                nn.Dropout(args.dropout_p),
                nn.Linear(encoder_dim // 2, args.num_cls)
            )
        else:
            self.classifier = None
            
        # ACF decoder
        if hasattr(args, 'acf_length') and args.acf_length > 0:
            self.acf_decoder = ACFDecoder(
                input_dim=encoder_dim,
                acf_length=args.acf_length,
                hidden_dims=getattr(args, 'acf_decoder_dims', [512, 256, 128])
            )
        else:
            self.acf_decoder = None

    def forward(self, x1, x2, y=None, meta=None):
        import time
        total_start = time.time()
        
        # Forward through SimSiam
        simsiam_start = time.time()
        out = self.simsiam(x1, x2)
        simsiam_time = time.time() - simsiam_start
        
        # Concatenate embeddings
        concat_start = time.time()
        z = torch.cat([out['z1'], out['z2']], dim=1)
        concat_time = time.time() - concat_start
        
        # Regression
        reg_start = time.time()
        output_reg = self.regressor(z)
        reg_time = time.time() - reg_start
        
        # Prepare output dictionary
        result = {
            'contrastive_loss': out['contrastive_loss'],
            'z1': out['z1'],
            'z2': out['z2'],
            'z': z,
            'regression': output_reg
        }
        
        # Classification
        cls_start = time.time()
        if self.classifier is not None:
            result['classification'] = self.classifier(z)
        cls_time = time.time() - cls_start
            
        # ACF reconstruction
        acf_start = time.time()
        if self.acf_decoder is not None:
            result['acf_reconstruction'] = self.acf_decoder(z)
        acf_time = time.time() - acf_start
        
        total_time = time.time() - total_start
        
        # Print timing info every 100 forward passes
        if hasattr(self, '_forward_count'):
            self._forward_count += 1
        else:
            self._forward_count = 1
            
        if self._forward_count % 100 == 0:
            print(f"[MODEL TIMING] Total: {total_time:.4f}s")
            print(f"  - SimSiam: {simsiam_time:.4f}s ({simsiam_time/total_time*100:.1f}%)")
            print(f"  - Concat: {concat_time:.4f}s ({concat_time/total_time*100:.1f}%)")
            print(f"  - Regression: {reg_time:.4f}s ({reg_time/total_time*100:.1f}%)")
            print(f"  - Classification: {cls_time:.4f}s ({cls_time/total_time*100:.1f}%)")
            print(f"  - ACF Decoder: {acf_time:.4f}s ({acf_time/total_time*100:.1f}%)")
            
        return result

class MultiTaskSimSiam(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        
        self.activation = nn.GELU()
        self.simsiam = SimSiam(encoder) 
        encoder_dim = self.simsiam.output_dim * 2
        self.output_dim = encoder_dim
        # print("encoder_dim: ", encoder_dim, 'conformer_encoder: ', conformer_args.encoder_dim)
        # Main regressor for concatenated embeddings (normal mode)
        self.regressor = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim//2),
            nn.BatchNorm1d(self.output_dim//2),
            self.activation,
            nn.Dropout(args.dropout_p),
            nn.Linear(self.output_dim//2, args.output_dim*args.num_quantiles)
        )
        
        # Individual regressors for neighbor pairs (separate predictions)
        individual_dim = self.simsiam.output_dim  # Single embedding dimension
        self.individual_regressor = nn.Sequential(
            nn.Linear(individual_dim, individual_dim//2),
            nn.BatchNorm1d(individual_dim//2),
            self.activation,
            nn.Dropout(args.dropout_p),
            nn.Linear(individual_dim//2, args.output_dim*args.num_quantiles)
        )
        if args.num_cls:
            self.classifier = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim//2),
                nn.BatchNorm1d(self.output_dim//2),
                self.activation,
                nn.Dropout(args.dropout_p),
                nn.Linear(self.output_dim//2, args.num_cls)
            )
        else:
            self.classifier = None
            
        # Add ACF decoder if specified
        if hasattr(args, 'acf_length') and args.acf_length > 0:
            self.acf_decoder = ACFDecoder(
                input_dim=self.output_dim,
                acf_length=args.acf_length,
                hidden_dims=getattr(args, 'acf_decoder_dims', [512, 256, 128])
            )
        else:
            self.acf_decoder = None

    def forward(self, x1, x2=None, y=None, meta=None, time1=None, time2=None):
        out = self.simsiam(x1, x2, time1=time1, time2=time2)
        z1, z2 = out['z1'], out['z2']
        
        # Check if we have neighbor pair targets (separate targets for each view)
        is_neighbor_pair = isinstance(y, dict) and 'anchor' in y and 'neighbor' in y
        
        # Always use individual regressor approach to ensure consistent output shapes across ranks
        output_reg_1 = self.individual_regressor(z1)  # Prediction for view 1
        output_reg_2 = self.individual_regressor(z2)  # Prediction for view 2
        
        if is_neighbor_pair:
            # For neighbor pairs: use both predictions as separate targets
            output_reg = torch.cat([output_reg_1, output_reg_2], dim=1)
            # For classification, use anchor view (z1) as primary
            combined_z = z1
        else:
            # Normal case: use average of both predictions for single target
            output_reg = 0.5 * (output_reg_1 + output_reg_2)
            # For classification, concatenate embeddings as before
            combined_z = torch.cat([z1, z2], dim=1)
        
        out['z'] = combined_z
        out['regression'] = output_reg
        
        if self.classifier is not None:
            out['classification'] = self.classifier(combined_z)
            
        if self.acf_decoder is not None:
            print("using acf decoder!!!")
            out['acf_reconstruction'] = self.acf_decoder(combined_z)
        return out

class MultiTaskVicReg(nn.Module):
    def __init__(self, encoder, args):
        super().__init__()
        
        self.activation = nn.GELU()
        
        # VicReg components
        self.encoder = encoder
        
        # Get encoder output dimension
        # You might need to adjust this based on your encoder's actual output
        if hasattr(encoder, 'output_dim'):
            encoder_output_dim = encoder.output_dim
        elif hasattr(encoder, 'num_features'):
            encoder_output_dim = encoder.num_features
        else:
            # Fallback - you may need to adjust this
            # encoder_output_dim = encoder.fc.in_features if hasattr(encoder, 'fc') else 512
            raise ValueError ('encoder has no output_dim attribute')
        
        # projector_dim = encoder_output_dim * 4
        
        # Projector for VicReg (similar to SimSiam's predictor)
        self.projector = nn.Sequential(
            nn.Linear(encoder_output_dim, args.projector_dim),
            nn.LayerNorm(args.projector_dim),
            nn.SiLU(),
            nn.Linear(args.projector_dim, args.projector_dim),
        )
        
        # Store output dimensions
        self.output_dim = args.projector_dim
        
        # VicReg loss hyperparameters
        # self.vicreg_lambda = getattr(args, 'vicreg_lambda', 1.0)  # invariance weight
        # self.vicreg_mu = getattr(args, 'vicreg_mu', 1.0)  # variance weight
        # self.vicreg_nu = getattr(args, 'vicreg_nu', 1.0)  # covariance weight
        self.vicreg_variance_eps = getattr(args, 'vicreg_variance_eps', 1e-4)
        
        # Main regressor for concatenated embeddings (kept for compatibility but not used)
        self.regressor = nn.Sequential(
            nn.Linear(args.projector_dim, args.projector_dim//2),
            nn.LayerNorm(args.projector_dim//2),
            nn.SiLU(),
            nn.Dropout(args.dropout_p),
            nn.Linear(args.projector_dim//2, args.output_dim*args.num_quantiles)
        )
        
        # Individual regressors for single embeddings
        self.individual_regressor = nn.Sequential(
             nn.Linear(args.projector_dim, args.projector_dim//2),
            nn.LayerNorm(args.projector_dim//2),
            nn.SiLU(),
            nn.Dropout(args.dropout_p),
            nn.Linear(args.projector_dim//2, args.output_dim*args.num_quantiles)
        )
        
        # Classification head (optional)
        if args.num_cls:
            self.classifier = nn.Sequential(
                nn.Linear(args.projector_dim, args.projector_dim//2),
                nn.LayerNorm(args.projector_dim//2),
                nn.SiLU(),
                nn.Dropout(args.dropout_p),
                nn.Linear(args.projector_dim//2, args.num_cls)
            )
        else:
            self.classifier = None
            
        # ACF decoder (optional)
        if hasattr(args, 'acf_length') and args.acf_length > 0:
            self.acf_decoder = ACFDecoder(
                input_dim=args.projector_dim,
                acf_length=args.acf_length,
                hidden_dims=getattr(args, 'acf_decoder_dims', [512, 256, 128])
            )
        else:
            self.acf_decoder = None
    
    def _vicreg_variance(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(0) <= 1:
            return torch.zeros(1, device=z.device, dtype=z.dtype)
        std = torch.sqrt(z.var(dim=0) + self.vicreg_variance_eps)
        penalty = F.relu(1.0 - std)
        return penalty.mean()
    
    def _vicreg_covariance(self, z: torch.Tensor) -> torch.Tensor:
        if z.size(0) <= 1:
            return torch.zeros(1, device=z.device, dtype=z.dtype)
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.size(0) - 1)
        diag = torch.diagonal(cov)
        cov = cov - torch.diag(diag)
        return cov.pow(2).sum() / z.size(1)
    
    def _vicreg_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        invariance = F.mse_loss(z1, z2)
        variance = self._vicreg_variance(z1) + self._vicreg_variance(z2)
        covariance = self._vicreg_covariance(z1) + self._vicreg_covariance(z2)
        return (
            self.vicreg_lambda * invariance
            + self.vicreg_mu * variance
            + self.vicreg_nu * covariance
        )
    
    @staticmethod
    def _off_diagonal(tensor: torch.Tensor) -> torch.Tensor:
        dim = tensor.size(0)
        return tensor.flatten()[:-1].view(dim - 1, dim + 1)[:, 1:].reshape(-1)
    
    def forward(self, x1, x2=None, y=None, meta=None, return_all=True):
        # Encode both views
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        if isinstance(h1, tuple):
            h1 = h1[0]
        elif isinstance(h1, dict):
            h1 = h1['emb']
        if isinstance(h2, tuple):
            h2 = h2[0]
        elif isinstance(h2, dict):
            h2 = h2['emb']
        
        # Project to higher dimension for VicReg loss only
        z1 = self.projector(h1)
        z2 = self.projector(h2)
        
        # Calculate VicReg loss on projected features
        invariance = F.mse_loss(z1, z2)
        variance = self._vicreg_variance(z1) + self._vicreg_variance(z2)
        covariance = self._vicreg_covariance(z1) + self._vicreg_covariance(z2)

        # Always use individual regressor approach to ensure consistent output shapes across ranks
        output_reg_1 = self.individual_regressor(z1)  # Prediction for view 1
        output_reg_2 = self.individual_regressor(z2)  # Prediction for view 2

        # Normal case: use average of both predictions for single target
        output_reg = 0.5 * (output_reg_1 + output_reg_2)
        # For classification, concatenate embeddings as before
        combined_z = torch.cat([z1, z2], dim=1)
        
        # Prepare output dictionary
        out = {
            'z1': z1,
            'z2': z2,
            'z': combined_z,
            'regression': output_reg,
            'classification': self.classifier(combined_z) if self.classifier is not None else None,
            'invariance': invariance.cpu(),
            'variance': variance.cpu(),
            'covariance': covariance.cpu()
        }
        
        # ACF reconstruction (if needed)
        if self.acf_decoder is not None:
            out['acf_reconstruction'] = self.acf_decoder(combined_z)
        
        return out

class MultiResRegressor(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        self.model = MultiTaskRegressor(args, conformer_args)
    
    def forward(self, x, x_high, y=None):
        output_reg, output_dec = self.model(x)
        output_reg_high, output_dec_high = self.model(x_high)      
        return output_reg, output_dec, output_reg_high, output_dec_high

class DoubleInputRegressor(nn.Module):
    def __init__(self, encoder1, encoder2, args, mixer=nn.Identity(), rope=None):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.mixer = mixer
        self.rope = rope
        self.dims1 = self.encoder1.in_channels
        self.expected_in_channels = getattr(args, "in_channels", None)
        if self.expected_in_channels is not None:
            self.dims2 = max(self.expected_in_channels - self.dims1, 0)
        else:
            self.dims2 = None
        self.stacked_input = args.stacked_input
        self.num_quantiles = getattr(args, 'num_quantiles', 1)
        self.target_dim = args.output_dim
        if self.encoder1.avg_output: # if we pool the features the dimension is doubled
            self.output_dim = encoder1.output_dim + encoder2.output_dim
        else:
            self.output_dim = encoder1.output_dim
        # print("DoubleInputRegressor output_dim: ", self.output_dim, "encoder1: ", encoder1.output_dim, "encoder2: ", encoder2.output_dim)
        if not args.encoder_only:
            self.regressor = nn.Sequential(
                nn.Linear(self.output_dim, self.output_dim//2),
                nn.GELU(),
                nn.Dropout(p=0.3),
                nn.Linear(self.output_dim//2, args.output_dim*args.num_quantiles)
            )
        else:
            self.regressor = nn.Identity()

    def _split_inputs(self, x1, x2):
        """
        Ensure both encoders receive tensors with the expected channel counts.
        When stacked_input is enabled we prioritise splitting from x1 so that
        the SimSiam backbone can operate on a single augmented view.
        """
        if not self.stacked_input:
            return x1, x2

        # Always derive the primary split from x1 to stay compatible with SimSiam
        if x1 is None:
            raise ValueError("Expected stacked input tensor, but received None.")

        if self.dims1 > 0:
            x1_split = x1[:, -self.dims1:, :]
        else:
            x1_split = x1

        # Start with the remaining channels from x1
        remaining = x1[:, :-self.dims1, :] if self.dims1 else x1

        if x2 is None:
            x2_split = remaining
        else:
            x2_split = x2

        if self.dims2 is not None:
            expected = self.dims2
            if x2_split is None or x2_split.shape[1] == 0:
                # Fall back to remaining channels from x1
                x2_split = remaining[:, :expected, :]
            elif x2_split.shape[1] > expected:
                # Trim to the expected width (drop auxiliary channels such as ACF)
                x2_split = x2_split[:, :expected, :]
            elif x2_split.shape[1] < expected:
                # Attempt to recover missing channels from the stacked tensor
                if remaining.shape[1] >= expected:
                    x2_split = remaining[:, :expected, :]
                else:
                    pad = torch.zeros(
                        x2_split.size(0),
                        expected - x2_split.shape[1],
                        x2_split.size(2),
                        device=x2_split.device,
                        dtype=x2_split.dtype,
                    )
                    x2_split = torch.cat([x2_split, pad], dim=1)
        return x1_split, x2_split

        
    def forward(self, x1, x2=None, time=None, return_all=False):
        x1, x2 = self._split_inputs(x1, x2)
        x1 = self.encoder1(x1)
        if isinstance(x1, tuple):
            x1 = x1[0]
        
        # [NEW] Pass time to encoder2 (Astroconformer)
        # Check if encoder2 accepts time (it should if it's Astroconformer)
        # Using try/except or inspection might be safer but we control the code.
        # We assume encoder2 is Astroconformer.
        try:
            x2 = self.encoder2(x2, time=time)
        except TypeError:
            x2 = self.encoder2(x2)
            
        if isinstance(x2, tuple):
            x2 = x2[0]
        # Ensure both tensors have same number of dimensions before cat
        if x1.dim() == 2 and x2.dim() == 3:
            x1 = x1.unsqueeze(1)
        elif x1.dim() == 3 and x2.dim() == 2:
            x2 = x2.unsqueeze(1)
        x = torch.cat([x1.nan_to_num(0), x2.nan_to_num(0)], dim=1)
        if self.rope is not None:
            RoPE = self.rope(x, x.shape[1]).nan_to_num(0)
            x = self.mixer(x, RoPE)
        # print("doubleinputregressor shapes: ", x1.shape, x2.shape)
        x_reg = x if len(x.shape)==2 else x.mean(1)
        out = self.regressor(x_reg)
        # Only reshape if we have a real regressor (not Identity) and quantiles > 1
        if self.num_quantiles > 1 and not isinstance(self.regressor, nn.Identity):
            out = out.reshape(out.shape[0], self.target_dim, self.num_quantiles)
        if return_all:
            return {
                'regression': out,
                'tokens': x,
                'encoding_1': x1,
                'encoding_2': x2
            }
        return out, x

class MultiEncoder(nn.Module):
    def __init__(self, args, conformer_args):
        super().__init__()
        if args.backbone == 'lstm':
            self.backbone = LSTMEncoder(args)
        else:
            self.backbone = CNNEncoder(args)
        self.head_size = conformer_args.encoder_dim // conformer_args.num_heads
        self.rotary_ndims = int(self.head_size * 0.5)
        self.pe = RotaryEmbedding(self.rotary_ndims)
        self.encoder = ConformerEncoder(conformer_args)
        self.output_dim = conformer_args.encoder_dim
        # self.avg_output = args.avg_output
        
    def forward(self, x, time=None, **kwargs):
        backbone_out = self.backbone(x)
        if len(backbone_out.shape) == 2:
            x_enc = backbone_out.unsqueeze(1)
        else:
            x_enc = backbone_out
        RoPE = self.pe(x_enc, x_enc.shape[1]).nan_to_num(0)
        x_enc = self.encoder(x_enc, RoPE)
        if (len(x_enc.shape) == 3):
            x_enc = x_enc.sum(dim=1)
        return x_enc, backbone_out


class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, args):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.attn = Flash_Mha(embed_dim=args.encoder_dim, num_heads=args.num_heads, dropout=args.dropout)
        self.ffn = MLP(in_features=args.encoder_dim)
        self.attn_norm = RMSNorm(args.encoder_dim)
        self.ffn_norm = RMSNorm(args.encoder_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Set first layer flag for initialization purposes
        self.encoder = nn.Linear(args.in_channels, args.encoder_dim)
        self.encoder.is_first_layer = True
        self.pooling_method=args.pooling_method
        
        # Create transformer blocks
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.num_layers):
            self.layers.append(Block(args))
        
        self.norm = RMSNorm(args.encoder_dim)
        self.head = MLP(args.encoder_dim, out_features=args.output_dim*args.num_quantiles, dtype=torch.get_default_dtype())
        self.output_dim = args.output_dim*args.num_quantiles
        
        # Calculate and store scaling factors for DeepNorm if needed
        if getattr(args, 'deepnorm', False) and args.num_layers >= 6:
            layer_coeff = args.num_layers / 6.0
            self.alpha = layer_coeff ** 0.5  
            self.beta = layer_coeff ** -0.5
        else:
            self.alpha = 1.0
            self.beta = 1.0
    def forward(self, x, y=None):
        if len(x.shape)==2:
            x = x.unsqueeze(-1)
        elif len(x.shape)==3 and x.shape[1]==1:
            x = x.permute(0,2,1)
        h = self.encoder(x)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if self.pooling_method == 'sum':
            h = h.sum(dim=1)
        elif self.pooling_method == 'cls_token':
            h = h[:,0,:]
        elif self.pooling_method == 'mean':
            h = h.mean(dim=1)
        output = self.head(h)        
        return output, y, h


class LSTMFeatureExtractor(nn.Module):
    def __init__(self, seq_len=1024, hidden_size=256, num_layers=5, num_classes=4,
                 in_channels=1, channels=256, dropout=0.2, kernel_size=4 ,stride=4, image=False):
        super(LSTMFeatureExtractor, self).__init__()
        self.seq_len = seq_len
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.image = image
        self.stride = stride
        self.t_features = self.seq_len//self.stride
        print("image: ", image)
        # self.conv = Conv2dSubampling(in_channels=in_channels, out_channels=channels)
        if not image:
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding='same', stride=1)
            self.pool = nn.MaxPool1d(kernel_size=stride)
            self.skip = nn.Conv1d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
            self.drop = nn.Dropout1d(p=dropout)
            self.batchnorm1 = nn.BatchNorm1d(channels)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding='same', stride=1)
            self.pool = nn.MaxPool2d(kernel_size=stride)
            self.skip = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, padding=0, stride=stride)
            self.drop = nn.Dropout2d(p=dropout)
            self.batchnorm1 = nn.BatchNorm2d(channels)        
             
        self.lstm = nn.LSTM(channels, hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
       
        self.activation = Sine()
        self.num_features = self._out_shape()
        self.output_dim = self.num_features

    def _out_shape(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        # Make sure to not mess up the random state.
        rng_state = torch.get_rng_state()
        # try:
        # print("calculating out shape")
        if not self.image:
            dummy_input = torch.randn(2,self.in_channels, self.seq_len)
        else:
            dummy_input = torch.randn(2,self.in_channels, self.seq_len, self.seq_len)
        # dummy_input = torch.randn(2,self.seq_len, self.in_channels)
        input_length = torch.ones(2, dtype=torch.int64)*self.seq_len
        # print("dummy_input: ", dummy_input.shape)
        # x = self.conv_pre(dummy_input)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(dummy_input)))))
        # x = self.conv(dummy_input, input_length)
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2)
        x_f,(h_f,_) = self.lstm(x)
        h_f = h_f.transpose(0,1).transpose(1,2)
        h_f = h_f.reshape(h_f.shape[0], -1)
        # print("finished")
        return h_f.shape[1] 
        # finally:
        #     torch.set_rng_state(rng_state)

    def forward(self, x, return_cell=False):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        elif len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.transpose(-1,-2)
        skip = self.skip(x)
        x = self.drop(self.pool(self.activation(self.batchnorm1(self.conv1(x)))))
        x = x + skip[:, :, :x.shape[-1]] # [B, C, L//stride]
        x = x.view(x.shape[0], x.shape[1], -1).swapaxes(1,2) # [B, L//stride, C]
        x_f,(h_f,c_f) = self.lstm(x) # [B, L//stride, 2*hidden_size], [B, 2*num_layers, hidden_size], [B, 2*num_layers, hidden_size
        if return_cell:
            return x_f, h_f, c_f
        h_f = h_f.transpose(0,1).transpose(1,2)
        h_f = h_f.reshape(h_f.shape[0], -1)
        return h_f


class LSTM_DUAL_LEGACY(nn.Module):
    def __init__(self, dual_model, encoder_dims, lstm_args, predict_size=128,
                 num_classes=4, num_quantiles=1, freeze=False, ssl=False, **kwargs):
        super(LSTM_DUAL_LEGACY, self).__init__(**kwargs)
        # print("intializing dual model")
        # if lstm_model is not None:
        self.feature_extractor = LSTMFeatureExtractor(**lstm_args)
        self.ssl= ssl
        # self.attention = lstm_model.attention
        if freeze:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
                # for param in self.attention.parameters():
                #     param.requires_grad = False
        num_lstm_features = self.feature_extractor.hidden_size*2
        self.num_features = num_lstm_features + encoder_dims
        self.output_dim = self.num_features
        self.dual_model = dual_model
        self.pred_layer = nn.Sequential(
        nn.Linear(self.num_features, predict_size),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(predict_size,num_classes//2 * num_quantiles),)
        self.conf_layer = nn.Sequential(
        nn.Linear(16, 16),
        nn.GELU(),
        nn.Dropout(p=0.3),
        nn.Linear(16,num_classes//2),)
    def lstm_attention(self, query, keys, values):
        # Query = [BxQ]
        # Keys = [BxTxK]
        # Values = [BxTxV]
        # Outputs = a:[TxB], lin_comb:[BxV]
        # Here we assume q_dim == k_dim (dot product attention)
        scale = 1/(keys.size(-1) ** -0.5)
        query = query.unsqueeze(1) # [BxQ] -> [Bx1xQ]
        keys = keys.transpose(1,2) # [BxTxK] -> [BxKxT]
        energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
        energy = F.softmax(energy.mul_(scale), dim=2) # scale, normalize
        linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
        return linear_combination
    
    def forward(self, acf, x=None, acf_phr=None):
        if x is None:
            x, acf = acf[:,:-1,:], acf[:,-1,:]
        if len(acf.shape) == 2:
            acf = acf.unsqueeze(1)
        x = x.squeeze()
        # print("acf, x: ", acf.shape, x.shape)
        acf, h_f, c_f = self.feature_extractor(acf, return_cell=True) # [B, L//stride, 2*hidden_size], [B, 2*nlayers, hidden_szie], [B, 2*nlayers, hidden_Size]
        c_f = torch.cat([c_f[-1], c_f[-2]], dim=1) # [B, 2*hidden_szie]
        t_features = self.lstm_attention(c_f, acf, acf) # [B, 2*hidden_size]
        d_features, _ = self.dual_model(x) # [B, encoder_dims]
        # print("nans in features: ", torch.isnan(t_features).sum(), torch.isnan(d_features).sum())
        features = torch.cat([t_features, d_features], dim=1) # [B, 2*hidden_size + encoder_dims]
        if self.ssl:
            return features
        out = self.pred_layer(features)
        return out
       
        # if acf_phr is not None:
        #     phr = acf_phr.reshape(-1,1).float()
        # else:
        #     phr = torch.zeros(features.shape[0],1, device=features.device)
        # mean_features = torch.nn.functional.adaptive_avg_pool1d(features.unsqueeze(1), 16).squeeze(1)
        # mean_features += phr
        # conf = self.conf_layer(mean_features)
        # # print("shapes in models: ", out.shape, conf.shape)
        # return torch.cat([out, conf], dim=1)


