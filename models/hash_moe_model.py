import torch
import torch.nn as nn
from models.model_components import RMSNorm, MHA_RoPE, MoeHashV2Layer
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

        self.moe_ffn = MoeHashV2Layer(config.dim, config.num_experts)

    def forward(self, x, mapped_tokens):
        h = x + self.attn(self.attn_norm(x))
        out = h + self.moe_ffn(self.ffn_norm(h), mapped_tokens)
        return out

class MoeHashGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.embed = nn.Embedding(config.vocab_size, config.dim)

        self.blocks = nn.ModuleList([
            MoeHashBlock(config) for _ in range(config.num_layers)
        ])

        self.proj_out = nn.Linear(config.dim, config.vocab_size)

        self.register_buffer('hash_map', self.create_hash_map(config.vocab_size, config.num_experts))

    def create_hash_map(self, vocab_size, num_experts):
        token_list = torch.randperm(vocab_size)
        experts_for_token = torch.randint(0, num_experts, (vocab_size,))
        hash_map = torch.zeros_like(token_list)
        hash_map[token_list] = experts_for_token[token_list]
        return hash_map

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
            mapped_tokens = self.hash_map[x]
            
            x = self.embed(x)

            for block in self.blocks:
                x = block(x, mapped_tokens)

            return self.proj_out(x)
    
    @torch.inference_mode()
    def generate(self, prefix_text=None, max_len=100):
        tokens = torch.tensor([[1]]) if prefix_text is None else torch.tensor(self.sampler.tokenizer.encode([prefix_text]))
        
        for _ in range(max_len):    
            logits = self.forward(tokens)
            next_token = self.sampler(logits)
            tokens = torch.cat((tokens, next_token), dim=-1)

        text = self.sampler.tokenizer.decode(tokens.view(-1).tolist()) 
        return text