import torch
import torch.nn as nn
from models.model_components import RMSNorm, MultiHeadAttention, GatedFeedForward
from dataclasses import dataclass

@dataclass
class DenseConfig:
    vocab_size: int
    batch_size: int
    cntx: int
    dim: int
    num_heads: int
    num_layers: int

class DenseBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(config.dim)
        self.attn = MultiHeadAttention(config.dim, config.num_heads)

        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = GatedFeedForward(config.dim)

    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        out = h + self.ffn(self.ffn_norm(x))
        return out

class DenseGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos = nn.Embedding(config.cntx, config.dim)

        self.blocks = nn.ModuleList([
            DenseBlock(config=config) for _ in range(config.num_layers)
        ])

        self.proj_out = nn.Linear(config.dim, config.vocab_size)

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, x):
        x = self.embed(x)
        x = x + self.pos(torch.arange(x.size(1), device=x.device))

        for block in self.blocks:
            x = block(x)

        return self.proj_out(x)