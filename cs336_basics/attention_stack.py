import torch
from torch import nn
from .layers import Linear


def softmax(x, dim=-1):
    max_x = x.max(dim=dim, keepdim=True).values
    shifted = x - max_x
    exp_x = torch.exp(shifted)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    probs = softmax(scores, dim=-1)
    out = probs @ v
    return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta, d_k, max_seq_len, device=None):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        positions = torch.arange(max_seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x, token_positions):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
        
        cos_pos = cos_pos.to(x.dtype)
        sin_pos = sin_pos.to(x.dtype)
        
        x_rot_even = x_even * cos_pos - x_odd * sin_pos
        x_rot_odd = x_even * sin_pos + x_odd * cos_pos
        
        out = torch.empty_like(x)
        out[..., 0::2] = x_rot_even
        out[..., 1::2] = x_rot_odd
        
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, rope_theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = RotaryPositionalEmbedding(
            theta=rope_theta,
            d_k=self.head_dim,
            max_seq_len=max_seq_len,
            device=device,
        )

    def _split_heads(self, x):
        b, s, _ = x.shape
        x = x.view(b, s, self.n_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        b, h, s, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, s, h * d)
        return x

    def _build_causal_mask(self, seq_len, device):
        base = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = base.view(1, 1, seq_len, seq_len)
        return mask

    def forward(self, x, token_positions):
        b, s, _ = x.shape
        device = x.device

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        pos = token_positions.unsqueeze(1)
        q = self.rope(q, pos)
        k = self.rope(k, pos)

        mask = self._build_causal_mask(s, device=device)

        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        out = self._merge_heads(attn_out)
        out = self.w_o(out)
        return out


class MultiHeadSelfAttentionNoRoPE(nn.Module):
    def __init__(self, d_model, n_heads, device=None, dtype=None):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.w_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.w_o = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, x):
        b, s, _ = x.shape
        x = x.view(b, s, self.n_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)
        return x

    def _merge_heads(self, x):
        b, h, s, d = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(b, s, h * d)
        return x

    def _build_causal_mask(self, seq_len, device):
        base = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        mask = base.view(1, 1, seq_len, seq_len)
        return mask

    def forward(self, x):
        b, s, _ = x.shape
        device = x.device

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        mask = self._build_causal_mask(s, device=device)

        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)

        out = self._merge_heads(attn_out)
        out = self.w_o(out)
        return out
