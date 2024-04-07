import torch.nn as nn
from .model_components import MoeRegLayer, RMSNorm, MHA_RoPE

class MoeRegBlock(nn.Module):
    def __init__(self, cntx, dim, num_heads, num_experts):
        super().__init__()

        self.attn_norm = RMSNorm(dim)
        self.attn = MHA_RoPE(cntx, dim, num_heads)

        self.ffn_norm = RMSNorm(dim)

        self.moe_ffn = MoeRegLayer(dim, num_experts)

    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        out = h + self.moe_ffn(self.ffn_norm(h))
        return out

class MoeRegGPT(nn.Module):
    def __init__(self, vocab_size, cntx, dim, num_heads, num_layers, num_experts):
        super().__init__()
        
        self.embed = nn.Embedding(vocab_size, dim)

        self.blocks = nn.ModuleList([
            MoeRegBlock(cntx, dim, num_heads, num_experts) for _ in range(num_layers)
        ])

        self.proj_out = nn.Linear(dim, vocab_size)

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        
        x = self.embed(x)

        for block in self.blocks:
            x = block(x)

        return self.proj_out(x)