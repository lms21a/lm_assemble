import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.model_components import RMSNorm, MHA_RoPE, GatedMLP
from dataclasses import dataclass

@dataclass
class LlamaConfig:
    dim: int
    max_cntx: int
    num_heads: int
    act_fn: nn.Module
    expansion_factor: int
    num_layers: int
    vocab_size: int

class LlamaKindaBlock(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()

        self.mha_norm = RMSNorm(config.dim)
        self.mha = MHA_RoPE(config.max_cntx, config.dim, config.num_heads)

        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = GatedMLP(config.dim, config.act_fn, config.expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.mha(self.mha_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out

class LlamaKinda(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        self.blocks = nn.ModuleList([
            LlamaKindaBlock(config=config) for _ in range(config.num_layers)
        ])

        self.proj_out = nn.Linear(config.dim, config.vocab_size)
        
    def forward(self, x):
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        return self.proj_out(x)
    
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        formatted_size = "{:,}".format(total_params)
        print(f"Model size: {formatted_size} parameters")

def get_llama_config(model_size: str):
    if model_size == 'tiny':
        return LlamaConfig(
            dim=64,
            max_cntx=32,
            num_heads=8,
            act_fn=F.gelu,
            expansion_factor=3,
            num_layers=4,
            vocab_size=8000
        )
    
    elif model_size == 'small':
        return LlamaConfig(
            dim=128,
            max_cntx=64,
            num_heads=16,
            num_layers=8,
            act_fn=F.gelu,
            expansion_factor=3,
            vocab_size=8000
        )