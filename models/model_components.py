from typing import Optional, Callable

import torch
from torch import einsum, Tensor
import torch.nn as nn
import torch.nn.functional as F

import einx

import math

# -----------------------------Convolution Layers-------------------------------
class Downsample(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.down = nn.Conv2d(channel, channel, kernel_size=3, stride=2)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.down(x)

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_groups=4):
        super().__init__()

        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.conv1(h * F.silu(h))

        h = self.norm2(h)
        h = self.conv2(h * F.silu(h))

        return h + (self.shortcut(x) if hasattr(self, "shortcut") else x)

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
    # Adopted From https://github.com/facebookresearch/llama/blob/main/llama/model.py
    _cache = {}
    def __init__(self, seq_len, head_size, base=10000):
        super().__init__()
        
        self.seq_len = seq_len
        self.head_size = head_size
        self.base = base

        # Key for caching
        key = (seq_len, head_size, base)
        if key not in RoPE._cache:
            RoPE._cache[key] = self._precompute_freqs_cis()
            self.register_buffer('freqs_cis', RoPE._cache[key])
            
        else:
            self.register_buffer('freqs_cis', RoPE._cache[key]) 
            
    def _precompute_freqs_cis(self):
        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_size, 2)[: (self.head_size // 2)].float() / self.head_size))
        t = torch.arange(self.seq_len, device=freqs.device)  # type: ignore
        freqs = torch.outer(t, freqs).float()  # type: ignore
        return torch.polar(torch.ones_like(freqs), freqs)  # complex64
    
    def _reshape_for_broadcast(self, freqs_cis, x):
        seq_len = x.size(1)  
        ndim = x.ndim
        assert 0 <= 1 < ndim
        freqs_cis_sliced = freqs_cis[:seq_len, :]
        assert freqs_cis_sliced.shape == (seq_len, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis_sliced.view(*shape)

    def apply_freq_cis(self, xq, xk):

        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = self._reshape_for_broadcast(self.freqs_cis, xq_)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

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
    
# TODO: Rename Attention Mechs For Clarity
class UnmaskedMHA(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        assert dim % num_heads == 0, 'Hidden Dimension needs to be divisible by num_heads'
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

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
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

        q, k = self.rope.apply_freq_cis(q, k)

        q, k, v = map(lambda x: x.transpose(1,2), (q,k,v))

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1,2).contiguous().view(B, T, C)
        
        return self.proj_out(y)
    
class ConvAttention(nn.Module):
    def __init__(self, in_channels, num_groups=32):
        super().__init__()
        assert in_channels % num_groups == 0, "Channels must be divisible by groups"

        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)

        self.wq = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.wk = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.wv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.wo = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        b, c, h, w = x.shape
        h_ = x
        q, k, v = self.wq(h_), self.wk(h_), self.wv(h_)

        q, k, v = map(lambda x: x.view(b, c, h*w).transpose(1,2), (q,k,v))

        h_ = F.scaled_dot_product_attention(q, k, v, attn_mask=None)
        h_ = h_.view(b, c, h, w)

        return self.wo(h_) + x

# Adapted From https://github.com/lucidrains/local-attention

class CausalLocalAttention(nn.Module):
    """
    Causal Local Attention
    Expects x: (batch_size seq_len, number_heads, head_dim)
    """

    def __init__(
        self,
        head_dim: int,
        cntx: int,
        window_size: int,
        look_backward: int = 1,
        use_flash_attn: bool = False,
        scale: Optional[float] = None,
    ):
        super().__init__()

        if use_flash_attn:
            print("WARNING: AT THE MOMENT, FLASH ATTENTION INTRODUCES NUMERICAL INSTABILITES")
            print('PROCEED WITH CAUTION')
        
        self.head_dim = head_dim
        self.cntx = cntx
        self.use_flash_attn = use_flash_attn

        self.scale = scale if scale is not None else head_dim ** -.5
        self.window_size = window_size

        self.look_backward = look_backward

        # relative positions
        self.rope = RoPE(cntx, head_dim)

    def max_neg_value(self, tensor):
        return -torch.finfo(tensor.dtype).max   

    def look_around(self, x, backward = 1, forward = 0, pad_value = -1, dim = 2):
        t = x.shape[1]
        dims = (len(x.shape) - dim) * (0, 0)
        padded_x = F.pad(x, (*dims, backward, forward), value = pad_value)
        tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
        return torch.cat(tensors, dim = dim)

    def forward(
            self,
            q: Tensor,
            k: Tensor,
            v: Tensor
    ) -> Tensor:

        shape, pad_value, window_size, look_backward, = q.shape, -1, self.window_size, self.look_backward

        # Pack for ease of shape use
        q, k, v = map(lambda x: einx.rearrange('b t h d -> (b h) t d', x, b=shape[0]), (q,k,v))

        b, n, dim_head, device= *q.shape, q.device


        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'

        windows = n // window_size

        seq = torch.arange(n, device = device)
        b_t = einx.rearrange('(w n) -> 1 w n', seq, w = windows, n = window_size)

        # bucketing
        
        bq, bk, bv = map(lambda t: einx.rearrange('b (w n) d -> b w n d', t, w = windows), (q, k, v))

        bq = bq * self.scale

        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  0,
            pad_value = pad_value
        )

        bk = self.look_around(bk, **look_around_kwargs)
        bv = self.look_around(bv, **look_around_kwargs)

        # rotary embeddings
        bq, bk = self.rope.apply_freq_cis(bq, bk)

        bq_t = b_t
        bq_k = self.look_around(b_t, **look_around_kwargs)

        bq_t = einx.rearrange('... i -> ... i 1', bq_t)
        bq_k = einx.rearrange('... j -> ... 1 j', bq_k)

        pad_mask = bq_k == pad_value

        causal_mask = bq_t < bq_k

        if self.use_flash_attn:
            mask = causal_mask | pad_mask
            out = F.scaled_dot_product_attention(bq, bk, bv, attn_mask=mask, scale=self.scale)
        
        else:    
            sim = einsum('b h i e, b h j e -> b h i j', bq, bk)
            mask_value = self.max_neg_value(sim)

            sim = sim.masked_fill(causal_mask, mask_value)
            del causal_mask

            sim = sim.masked_fill(pad_mask, mask_value)

            # attention

            attn = sim.softmax(dim = -1)

            # aggregation
            out = einsum('b h i j, b h j e -> b h i e', attn, bv)

        out = einx.rearrange('(b h) w n c -> b (w n) h c', out, w=windows, b=shape[0])

        return out

# Adapted from https://arxiv.org/abs/2402.19427
class LocalMQA(nn.Module):
    def __init__(
            self,
            dim: int,
            cntx: int,
            qheads: int,
            window_size: int
    ):
        super().__init__()
        
        assert dim % qheads == 0, f'Embedding Dimension ({dim}) must be divisible by the number of query heads {qheads}'
        head_dim = dim // qheads

        self.qheads = qheads

        self.wq = nn.Linear(dim, qheads * head_dim, bias=False)
        self.wkv = nn.Linear(dim, head_dim * 2, bias=False)

        self.local_attn = CausalLocalAttention(head_dim, cntx, window_size)
    
    def forward(self, x):
        q = einx.rearrange('b t (h c) -> b t h c', self.wq(x), h = self.qheads)

        k, v = einx.rearrange('b t ((h c) + (h c)) -> b t h c, b t h c', self.wkv(x), h=1)

        k,v = map(lambda x: einx.rearrange('b t h c -> b t (h r) c', x, r=self.qheads), (k, v))

        y = self.local_attn(q, k, v)

        y = einx.rearrange('b t h c -> b t (h c)', y)

        return y

# -----------------------------Feed Forward Blocks-------------------------------

class GatedFeedForward(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.proj_in = nn.Linear(dim, dim * 4)
        self.gate = nn.Linear(dim, dim * 4)
        self.proj_out = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.proj_out(F.silu(self.gate(x)) * self.proj_in(x))

class GatedMLP(nn.Module):
    def __init__(self, dim: int, act_fn: Callable[[torch.Tensor], torch.Tensor] = F.gelu, expansion_factor: int = 3):
        super().__init__()

        self.gate = nn.Linear(dim, dim*expansion_factor)
        self.act_fn = act_fn

        self.w1 = nn.Linear(dim, dim*expansion_factor)
        self.wo = nn.Linear(dim * expansion_factor, dim)
    
    def forward(self, x):
        return self.wo(self.act_fn(self.gate(x)) * self.w1(x))
    
# -----------------------------Mixture of Expert Layers-------------------------------
    
class MoeHashLayer(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        
        self.num_experts = num_experts
        self.experts = nn.ModuleList([GatedFeedForward(dim) for _ in range(num_experts)])

        self.rand_maps_cache = {} # After Training Can Save Cache as standalone dictionary
        self.cache_key = None

    def forward(self, x):
        B, T, C = x.shape
        current_key = (B, T)

        if self.cache_key != current_key:
            self.rand_maps_cache[current_key] = torch.randint(0, self.num_experts, (B * T,), dtype=torch.long)
            self.cache_key = current_key

        x = x.view(-1, C)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            idx = torch.where(self.rand_maps_cache[current_key] == i)[0]
            output[idx] += expert(x[idx])

        return output.view(B, T, C)

class MoeHashV2Layer(nn.Module):
    def __init__(self, dim, num_experts):
        super().__init__()
        
        self.num_experts = num_experts
        self.experts = nn.ModuleList([GatedFeedForward(dim) for _ in range(num_experts)])

    def forward(self, x, mapped_tokens):
        B, T, C = x.shape
        x = x.view(-1, C)

        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            idx = torch.where(mapped_tokens == i)[0]
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

# -----------------------------RNN Layers-------------------------------
# Adapted from https://arxiv.org/abs/2402.19427
class RG_LRU(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.input_gate = nn.Linear(dim, dim, bias=True)
        self.recurrent_gate = nn.Linear(dim, dim, bias=True)

        self.cap_lambda = nn.Parameter(torch.randn(1).uniform_(.9, .999))

    def calc_at(self, rt):
        lhs = -8 * F.softplus(self.cap_lambda)
        log_at = lhs * rt
        return torch.exp(log_at)
    
    def circline_transform(self, at):
        return torch.sqrt(1 - (at**2))
    
    def calc_ht(self, xt, ht_prev):
        it, rt = torch.sigmoid(self.input_gate(xt)), torch.sigmoid(self.recurrent_gate(xt))

        at = self.calc_at(rt)

        ht = (at * ht_prev) + (self.circline_transform(at) * (it * xt))
        return ht

    def forward(self, x):
        b, t, c = x.shape
        ht_prev = torch.zeros(b, c)
        states = []
        
        for i in range(t):
            xt = x[:, i, :]
            ht = self.calc_ht(xt, ht_prev)
            ht_prev = ht
            states.append(ht)
        
        yt = einx.rearrange('b (t c) -> b t c', torch.cat(states, 1), t=t)
        
        return yt
    
class RecurrentBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()

        self.gate = nn.Linear(dim, dim)
        
        self.fc_in = nn.Linear(dim, dim)

        # assuming these settings to maintain shape
        self.conv_layer = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=4, dilation=2, padding=3)

        self.cell = RG_LRU(dim)
        self.fc_out = nn.Linear(dim, dim)
    
    def forward(self, x):
        gate_out, temp_out = F.gelu(self.gate(x)), self.fc_in(x)

        temp_out = einx.rearrange('b t c -> b c t', temp_out)
        temp_out = einx.rearrange('b c t -> b t c',self.conv_layer(temp_out))

        temp_out = self.cell(temp_out)
        return self.fc_out(
            temp_out * gate_out
        )