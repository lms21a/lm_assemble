import torch.nn as nn
from models.model_components import RMSNorm, MHA_RoPE, MoeHashLayer
from dataclasses import dataclass

@dataclass
class MoeHashGPTConfig:
    vocab_size: int
    batch_size: int
    cntx: int
    dim: int
    num_heads: int
    num_layers: int
    num_experts: int

class MoeHashBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(config.dim)
        self.attn = MHA_RoPE(config.cntx, config.dim, config.num_heads)

        self.ffn_norm = RMSNorm(config.dim)

        self.moe_ffn = MoeHashLayer(config.batch_size, config.cntx, config.dim, config.num_experts)

    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        out = h + self.moe_ffn(self.ffn_norm(h))
        return out

class MoeHashGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        self.blocks = nn.ModuleList([
            MoeHashBlock(config) for _ in range(config.num_layers)
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