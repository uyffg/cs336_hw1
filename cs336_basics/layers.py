import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x):
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        mean_sq = (x * x).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        x_norm = x / rms
        y = x_norm * self.weight
        return y.to(orig_dtype)


def _make_d_ff(d_model):
    raw = int((8.0 / 3.0) * d_model)
    if raw % 64 != 0:
        raw = ((raw // 64) + 1) * 64
    return raw


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = _make_d_ff(d_model)
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x):
        x1 = self.w1(x)
        x3 = self.w3(x)
        gate = torch.nn.functional.silu(x1)
        hidden = gate * x3
        out = self.w2(hidden)
        return out
