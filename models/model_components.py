import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------------------Norms-----------------------------------------------
class RMSNorm(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

# -----------------------------Positional Encoding-------------------------------
    
class RoPE(nn.Module):
    # Adopted From https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
    _cache = {}
    def __init__(self, seq_len, head_size, base=10000):
        super().__init__()

        assert head_size % 2 == 0, 'Head Size must be even'
        
        self.seq_len = seq_len
        self.head_size = head_size
        self.base = base

        # Key for caching
        key = (seq_len, head_size, base)
        if key not in RoPE._cache:
            self._precompute_freqs_cis()
            RoPE._cache[key] = self.freqs_cis
            
        else:
            self.freqs_cis = RoPE._cache[key]            
            
    def _precompute_freqs_cis(self):

        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_size, 2)[: (self.head_size // 2)].float() / self.head_size))
        t = torch.arange(self.seq_len, device=freqs.device)
        freqs = torch.outer(t, freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.freqs_cis = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)

    def apply_freq_cis(self, x):
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2).to(x.device)
        freqs_cis = self.freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2).to(x.device)
        x_out2 = torch.stack(
            [
                xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
                xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
            ],
            -1,
        ).to(x.device)
    
        x_out2 = x_out2.flatten(3)
        return x_out2.type_as(x)

class SinPosEnc(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

    def forward(self, x):
        T = x.size(1)
        device = x.device
        
        position = torch.arange(T, dtype=torch.float, device=device).view(-1, 1)
    
        pe = torch.zeros(T, self.dim, device=device)
        pe[:, 0::2] = torch.sin(position * self.div_term)
        pe[:, 1::2] = torch.cos(position * self.div_term)
    
        return x + pe

# -----------------------------Attention Mechanisms-------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        
        self.head_size = self.dim // self.num_heads
        
        self.norm = RMSNorm(dim)
        self.wqkv = nn.Linear(self.dim, self.dim * 3)
        self.proj_out = nn.Linear(self.dim, self.dim)

    def forward(self, x):
        B, T, C = x.shape
        
        x = F.relu(self.norm(x))
        
        q, k, v = self.wqkv(x).split([self.dim, self.dim, self.dim], dim = -1)

        q, k, v = map(lambda x: x.view(B, T, self.num_heads, self.head_size).transpose(1,2), (q,k,v))

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
        return self.proj_out(y)
    
class MHA_RoPE(nn.Module):
    def __init__(self, cntx, dim, num_heads):
        super().__init__()
    
        self.dim = dim
        self.num_heads = num_heads
        self.head_size = self.dim // self.num_heads
        
        self.rope = RoPE(cntx, self.head_size)
        
        self.wqkv = nn.Linear(dim, dim * 3)
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, T, C = x.shape
        
        q, k, v = self.wqkv(x).split([self.dim, self.dim, self.dim], dim=-1)

        q, k, v = map(lambda x: x.view(B, T, self.num_heads, self.head_size), (q,k,v))

        q, k = self.rope.apply_freq_cis(q), self.rope.apply_freq_cis(k)

        q, k, v = map(lambda x: x.transpose(1,2), (q,k,v))

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
        return self.proj_out(y)

# -----------------------------Feed Forward Blocks-------------------------------

class GatedFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_in = nn.Linear(dim, dim * 4)
        self.gate = nn.Linear(dim, dim * 4)
        self.proj_out = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.proj_out(F.silu(self.gate(x)) * self.proj_in(x))
    
# -----------------------------Mixture of Expert Layers-------------------------------
    
class MoeHashLayer(nn.Module):
    def __init__(self, bsz, cntx, dim, num_experts):
        super().__init__()
        
        self.experts = nn.ModuleList([GatedFeedForward(dim) for _ in range(num_experts)])
        self.rand_maps = torch.randint(0, num_experts, (bsz * cntx,), dtype=torch.long)

    def forward(self, x):
            
        B, T, C = x.shape

        x = x.view(-1, C)
        
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            idx = torch.where(self.rand_maps == i)[0]
            output[idx] += expert(x[idx])

        return output.view(B, T, C)
    
class MoeRegLayer(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        self.experts = nn.ModuleList([GatedFeedForward(dim) for _ in range(num_experts)])
        self.gate = nn.Linear(dim, num_experts, bias=False)

    def forward(self, x):            
        B, T, C = x.shape
        x = x.view(B*T, C)
        
        logits = self.gate(x)
        hidden_states, idxs = torch.topk(logits, 2)
        hidden_states = F.softmax(hidden_states, dim=1)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            idx, chosen_expert = torch.where(idxs == i)
            output[idx] += hidden_states[idx, chosen_expert, None] * expert(x[idx])
            
        return output.view(B,T,C)