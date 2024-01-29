import torch.nn as nn
from models.model_components import RMSNorm, MHA_RoPE, GatedFeedForward
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
        self.attn = MHA_RoPE(config.cntx, config.dim, config.num_heads)

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

        for block in self.blocks:
            x = block(x)

        return self.proj_out(x)