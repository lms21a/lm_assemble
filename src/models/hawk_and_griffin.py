# Adapted from https://arxiv.org/abs/2402.19427

import torch
import torch.nn as nn

from src.models.model_components import RMSNorm, RecurrentBlock, GatedMLP, LocalMQA

from dataclasses import dataclass

# Simple implementation for Hawk
@dataclass
class HawkConfig:
    vocab_size: int
    dim: int
    cntx: int
    num_blocks: int

class HawkBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.norm1 = RMSNorm(dim)
        self.mixing_block = RecurrentBlock(dim)

        self.norm2 = RMSNorm(dim)
        self.ffn = GatedMLP(dim)

    def forward(self, x):
        h = x + self.mixing_block(self.norm1(x))
        out = h + self.ffn(self.norm2(h))
        return out

class Hawk(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleDict({
            f'layer_{i}': HawkBlock(config.dim)
            for i in range(config.num_blocks)
        })

        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight

    def forward(self, x):
        x = self.embed(x)

        for layer in self.layers.values():
            x = layer(x)

        x = self.lm_head(x)
        return x

    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        formatted_size = "{:,}".format(total_params)
        print(f"Model size: {formatted_size} parameters")


@dataclass
class GriffinConfig:
    vocab_size: int
    dim: int
    cntx: int
    qheads: int
    window_size: int
    num_blocks: int
    
# Will use alternating instead of 2v1 for simplicity sake
class GriffinBlock(nn.Module):
    def __init__(
            self,
            layer_id: int,
            dim: int,
            cntx: int,
            qheads: int,
            window_size: int
    ):
        super().__init__()
        
        self.norm1 = RMSNorm(dim)
        self.temporal_block = LocalMQA(dim, cntx, qheads, window_size) if layer_id % 2 == 0 else RecurrentBlock(dim)

        self.norm2 = RMSNorm(dim)
        self.ffn = GatedMLP(dim)

    def forward(self, x):
        h = x + self.temporal_block(self.norm1(x))
        out = h + self.ffn(self.norm2(h))
        return out

class Griffin(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size, config.dim)

        self.layers = nn.ModuleDict({
            f'layer_{i}': GriffinBlock(
                i,
                config.dim,
                config.cntx,
                config.qheads,
                config.window_size
            )
            for i in range(config.num_blocks)
        })

        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.embed.weight = self.lm_head.weight

    def forward(self, x):
        x = self.embed(x)

        for layer in self.layers.values():
            x = layer(x)
        
        x = self.lm_head(x)
        return x
        
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        formatted_size = "{:,}".format(total_params)
        print(f"Model size: {formatted_size} parameters")

        
