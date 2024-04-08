import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Callable

from src.models.model_components import MHA_RoPE, GatedMLP, RMSNorm, MoDBlock

# Adapted from https://arxiv.org/abs/2404.02258
# Excellent work from the deepmind team! Was able to replicate results

@dataclass
class MoDConfig:
    dim: int
    max_cntx: int
    num_heads: int
    act_fn: Callable[[torch.Tensor], torch.Tensor]
    expansion_factor: int
    num_layers: int
    cap_percentile: float # Capacity Percentile of tokens that are processed by the RegBlock
    vocab_size: int

class RegBlock(nn.Module):
    def __init__(self, config: MoDConfig):
        super().__init__()

        self.mha_norm = RMSNorm(config.dim)
        self.mha = MHA_RoPE(config.max_cntx, config.dim, config.num_heads)

        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = GatedMLP(config.dim, config.act_fn, config.expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.mha(self.mha_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out

class MoDTransformer(nn.Module):
    def __init__(self, config: MoDConfig):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size, config.dim)

        self.blocks = nn.ModuleList([
            MoDBlock(
                block=RegBlock(config),
                dim=config.dim,
                max_cntx=config.max_cntx,
                cap_percentile=config.cap_percentile
            )
            for _ in range(config.num_layers)
        ])

        self.proj_out = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)

        aux_loss = 0
        for block in self.blocks:
            x, aux_loss_per_block = block(x)
            aux_loss += aux_loss_per_block

        return self.proj_out(x), aux_loss
    
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        formatted_size = "{:,}".format(total_params)
        print(f"Model size: {formatted_size} parameters")

def get_mod_config(model_size: str):

    if model_size == 'tiny':
        return MoDConfig(
            dim=64,
            max_cntx=32,
            num_heads=8,
            act_fn=F.gelu,
            expansion_factor=3,
            num_layers=4,
            cap_percentile=.3,
            vocab_size=8000
        )
    
    elif model_size == 'small':
        return MoDConfig(
            dim=128,
            max_cntx=64,
            num_heads=16,
            act_fn=F.gelu,
            expansion_factor=3,
            num_layers=8,
            cap_percentile=.5,
            vocab_size=8000
        )