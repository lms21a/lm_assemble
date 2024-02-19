import torch
import torch.nn as nn
from models.model_components import RMSNorm, GatedFeedForward, UnmaskedMHA, MoeRegLayer
from dataclasses import dataclass
import numpy as np

@dataclass
class EncoderConfig:
    vocab_size: int
    batch_size: int
    cntx: int
    dim: int
    num_heads: int
    num_layers: int
    num_classes: int

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(config.dim)
        self.attn = UnmaskedMHA(config.dim, config.num_heads)

        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = GatedFeedForward(config.dim)

    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out

class EncoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        flattened_dim = config.dim * config.cntx

        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos = nn.Embedding(config.cntx, config.dim)

        self.blocks = nn.ModuleList([
            EncoderBlock(config=config) for _ in range(config.num_layers)
        ])

        self.pre_proj_out = nn.Linear(flattened_dim, flattened_dim // 4)
        self.proj_out = nn.Linear(flattened_dim // 4, config.num_classes)
        
    def forward(self, x):
        B, _ = x.shape
        x = self.embed(x)
        x = x + self.pos(torch.arange(x.size(1), device=x.device))

        for block in self.blocks:
            x = block(x)
        
        x = x.view(B, -1)
        x = self.pre_proj_out(x)
        
        return self.proj_out(x)

    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        formatted_size = "{:,}".format(total_params)
        print(f"Model size: {formatted_size} parameters")

@dataclass
class MoeEncoderConfig:
    vocab_size: int
    batch_size: int
    cntx: int
    dim: int
    num_heads: int
    num_layers: int
    num_experts: int
    num_classes: int

class MoeEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attn_norm = RMSNorm(config.dim)
        self.attn = UnmaskedMHA(config.dim, config.num_heads)

        self.ffn_norm = RMSNorm(config.dim)
        self.ffn = MoeRegLayer(dim=config.dim, num_experts=config.num_experts)

    def forward(self, x):
        h = x + self.attn(self.attn_norm(x))
        out = h + self.ffn(self.ffn_norm(h))
        return out

class MoeEncoderTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.pos = nn.Embedding(config.cntx, config.dim)

        self.blocks = nn.ModuleList([
            MoeEncoderBlock(config=config) for _ in range(config.num_layers)
        ])

        self.proj_out = nn.Linear(config.dim * config.cntx, config.num_classes)
        
    def forward(self, x):
        B, _ = x.shape
        x = self.embed(x)
        x = x + self.pos(torch.arange(x.size(1), device=x.device))

        for block in self.blocks:
            x = block(x)
        
        x = x.view(B, -1)
        
        return self.proj_out(x)

    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        formatted_size = "{:,}".format(total_params)
        print(f"Model size: {formatted_size} parameters")
    
    def _print_model_size(model):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"Model has {params} trainable parameters")