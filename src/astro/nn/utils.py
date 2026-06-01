"""
nn/utils.py — model loading utilities for the astrophysical cross-modal experiment.

Adapted from MultiDESA (original: nn/utils.py).
Changes vs original:
  - Container class inlined (no dependency on util.utils)
  - torchinfo install/import removed (only used in compare_model_architectures)
  - get_lightPred_model removed (server-specific hardcoded paths, unused here)
"""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingLR
from collections import OrderedDict
from .Modules.mhsa_pro import MHA_rotary
from .DualFormer.dual_attention import DualAttention
from .Modules.cnn import ConvBlock
from nn.models import *
from nn.moco import MultimodalMoCo
from nn.simsiam import SimCLR, SimSiam, MultiModalSimSiam
from nn.astroconf import Astroconformer, AstroEncoderDecoder
from nn.scheduler import WarmupScheduler
import yaml
import os


class Container(object):
    """A container class that can be used to store any attributes."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def load_dict(self, dict):
        for key, value in dict.items():
            if getattr(self, key, None) is None:
                setattr(self, key, value)

    def print_attributes(self):
        for key, value in vars(self).items():
            print(f"{key}: {value}")

    def get_dict(self):
        return self.__dict__


models = {'Astroconformer': Astroconformer, 'CNNEncoder': CNNEncoder, 'SimCLR': SimCLR, 'SimSiam': SimSiam,
          'MultiModalSimSiam': MultiModalSimSiam, 'MultimodalMoCo': MultimodalMoCo,
          'AstroEncoderDecoder': AstroEncoderDecoder,}

schedulers = {'WarmupScheduler': WarmupScheduler, 'OneCycleLR': OneCycleLR,
              'CosineAnnealingLR': CosineAnnealingLR, 'none': None}


def load_checkpoints_ddp(model, checkpoint_path, prefix='', load_backbone=False):
    print(f"****Loading  checkpoint - {checkpoint_path}****")
    state_dict = torch.load(f'{checkpoint_path}', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        while key.startswith('module.'):
            key = key[7:]
        if load_backbone:
            if key.startswith('backbone.'):
                key = key[9:]
            else:
                continue
        key = prefix + key
        new_state_dict[key] = value
    state_dict = new_state_dict

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("number of keys in state dict and model: ", len(state_dict), len(model.state_dict()))
    print("number of missing keys: ", len(missing))
    print("number of unexpected keys: ", len(unexpected))
    print("missing keys: ", missing)
    print("unexpected keys: ", unexpected)
    return model


def init_model(model, model_args, prefix='', load_backbone=False):
    if model_args.load_checkpoint:
        model = load_checkpoints_ddp(model, model_args.checkpoint_path, prefix=prefix, load_backbone=load_backbone)
    else:
        print("****applying deepnorm initialization****")
        if hasattr(model, 'encoder') and any(
            'transformer' in str(type(module)).lower()
            or 'conformer' in str(type(module)).lower()
            or 'mhsa' in str(type(module)).lower()
            for name, module in model.named_modules()):
            deepnorm_init(model, model_args)
    return model


def deepnorm_init(model, args):
    from nn.models import Transformer
    if isinstance(model, Transformer):
        beta = getattr(args, 'beta', 1)
        if hasattr(model, 'encoder'):
            nn.init.xavier_normal_(model.encoder.weight, gain=1)
            if model.encoder.bias is not None:
                nn.init.zeros_(model.encoder.bias)
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'attn'):
                if hasattr(layer.attn, 'query_proj'):
                    nn.init.xavier_normal_(layer.attn.query_proj.weight, gain=1)
                    nn.init.xavier_normal_(layer.attn.key_proj.weight, gain=1)
                    nn.init.xavier_normal_(layer.attn.value_proj.weight, gain=beta)
                    nn.init.xavier_normal_(layer.attn.output_proj.weight, gain=beta)
                    if hasattr(layer.attn.query_proj, 'bias') and layer.attn.query_proj.bias is not None:
                        nn.init.zeros_(layer.attn.query_proj.bias)
                        nn.init.zeros_(layer.attn.key_proj.bias)
                        nn.init.zeros_(layer.attn.value_proj.bias)
                        nn.init.zeros_(layer.attn.output_proj.bias)
            elif hasattr(layer, 'ffn'):
                nn.init.xavier_normal_(layer.ffn.fc1.weight, gain=beta)
                nn.init.xavier_normal_(layer.ffn.fc2.weight, gain=beta)
                if hasattr(layer.ffn.fc1, 'bias') and layer.ffn.fc1.bias is not None:
                    nn.init.zeros_(layer.ffn.fc1.bias)
                    nn.init.zeros_(layer.ffn.fc2.bias)
        if hasattr(model, 'head'):
            if hasattr(model.head, 'linear1'):
                nn.init.xavier_normal_(model.head.fc1.weight, gain=beta)
            if hasattr(model.head, 'linear2'):
                nn.init.xavier_normal_(model.head.fc2.weight, gain=beta)
        return

    def init_func(m):
        beta = getattr(args, 'beta', 1)
        if isinstance(m, MHA_rotary):
            nn.init.xavier_normal_(m.query.weight, gain=1)
            nn.init.xavier_normal_(m.key.weight, gain=1)
            nn.init.xavier_normal_(m.value.weight, gain=beta)
            nn.init.xavier_normal_(m.output.weight, gain=beta)
            nn.init.zeros_(m.query.bias)
            nn.init.zeros_(m.key.bias)
            nn.init.zeros_(m.value.bias)
            nn.init.zeros_(m.output.bias)
            if getattr(m, 'ffn', None) is not None:
                nn.init.xavier_normal_(m.ffn.linear1.weight, gain=beta)
                nn.init.xavier_normal_(m.ffn.linear2.weight, gain=beta)
                nn.init.zeros_(m.ffn.linear1.bias)
                nn.init.zeros_(m.ffn.linear2.bias)
        elif isinstance(m, DualAttention):
            nn.init.xavier_normal_(m.q1_proj.weight, gain=1)
            nn.init.xavier_normal_(m.k1_proj.weight, gain=1)
            nn.init.xavier_normal_(m.v1_proj.weight, gain=1)
            nn.init.xavier_normal_(m.q2_proj.weight, gain=1)
            nn.init.xavier_normal_(m.k2_proj.weight, gain=beta)
            nn.init.xavier_normal_(m.v2_proj.weight, gain=beta)
            nn.init.xavier_normal_(m.out1_proj.weight, gain=beta)
            nn.init.xavier_normal_(m.out2_proj.weight, gain=beta)
        elif hasattr(m, 'attn') and hasattr(m.attn, 'query_proj'):
            nn.init.xavier_normal_(m.attn.query_proj.weight, gain=1)
            nn.init.xavier_normal_(m.attn.key_proj.weight, gain=1)
            nn.init.xavier_normal_(m.attn.value_proj.weight, gain=beta)
            nn.init.xavier_normal_(m.attn.output_proj.weight, gain=beta)
            if hasattr(m.attn.query_proj, 'bias') and m.attn.query_proj.bias is not None:
                nn.init.zeros_(m.attn.query_proj.bias)
                nn.init.zeros_(m.attn.key_proj.bias)
                nn.init.zeros_(m.attn.value_proj.bias)
                nn.init.zeros_(m.attn.output_proj.bias)
        elif hasattr(m, 'ffn') and hasattr(m.ffn, 'linear1'):
            nn.init.xavier_normal_(m.ffn.linear1.weight, gain=beta)
            nn.init.xavier_normal_(m.ffn.linear2.weight, gain=beta)
            if hasattr(m.ffn.linear1, 'bias') and m.ffn.linear1.bias is not None:
                nn.init.zeros_(m.ffn.linear1.bias)
                nn.init.zeros_(m.ffn.linear2.bias)
        elif isinstance(m, nn.Linear):
            if getattr(m, 'is_first_layer', False):
                nn.init.xavier_normal_(m.weight, gain=1)
            else:
                nn.init.xavier_normal_(m.weight, gain=beta)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    model.apply(init_func)


def load_scheduler(optimizer, train_dataloader, world_size, optim_args, data_args):
    schedulers = {
        'OneCycleLR': OneCycleLR,
        'CosineAnnealingLR': CosineAnnealingLR,
        'WarmupScheduler': WarmupScheduler,
        'none': None
    }
    if optim_args.scheduler == 'none':
        return None
    try:
        scheduler_class = schedulers.get(optim_args.scheduler)
        if scheduler_class is None:
            print(f"Warning: Scheduler {optim_args.scheduler} not found.")
            return None
        scheduler_args = dict(optim_args.scheduler_args.get(optim_args.scheduler, {}))
        numeric_keys = [
            'max_lr', 'epochs', 'steps_per_epoch', 'pct_start',
            'base_momentum', 'max_momentum', 'div_factor', 'final_div_factor',
            'eta_min', 'T_max'
        ]
        for key in numeric_keys:
            if key in scheduler_args:
                try:
                    scheduler_args[key] = float(scheduler_args[key])
                except (ValueError, TypeError):
                    print(f"Warning: Could not convert {key} to float")
        scheduler_args['optimizer'] = optimizer
        if 'steps_per_epoch' in scheduler_args:
            scheduler_args['steps_per_epoch'] = len(train_dataloader) * world_size
        if 'epochs' in scheduler_args:
            scheduler_args['epochs'] = int(data_args.num_epochs)
        if optim_args.scheduler == 'OneCycleLR':
            if 'max_lr' not in scheduler_args:
                scheduler_args['max_lr'] = float(optim_args.max_lr)
            if 'steps_per_epoch' not in scheduler_args:
                scheduler_args['steps_per_epoch'] = len(train_dataloader) * world_size
            if 'epochs' not in scheduler_args:
                scheduler_args['epochs'] = int(data_args.num_epochs)
        elif optim_args.scheduler == 'CosineAnnealingLR':
            if 'T_max' not in scheduler_args:
                scheduler_args['T_max'] = int(len(train_dataloader) * world_size)
        scheduler = scheduler_class(**scheduler_args)
        print(f"Scheduler {optim_args.scheduler} initialized successfully")
        return scheduler
    except Exception as e:
        print(f"Error initializing scheduler {optim_args.scheduler}: {e}")
        return None
