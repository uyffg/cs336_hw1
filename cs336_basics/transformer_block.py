import torch
from torch import nn
from .layers import RMSNorm, SwiGLUFFN, Embedding, Linear
from .attention_stack import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, rope_theta, max_seq_len, device=None, dtype=None):
        super().__init__()

        self.norm1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.norm2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLUFFN(d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions):
        h = self.norm1(x)
        h = self.attn(h, token_positions)
        x = x + h

        h2 = self.norm2(x)
        h2 = self.ffn(h2)
        out = x + h2

        return out


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, device=None, dtype=None):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=num_heads,
                d_ff=d_ff,
                rope_theta=rope_theta,
                max_seq_len=context_length,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        
    def forward(self, in_indices, token_positions=None):
        b, s = in_indices.shape
        device = in_indices.device
        
        if token_positions is None:
            token_positions = torch.arange(s, device=device).unsqueeze(0).expand(b, -1)
        
        x = self.token_embeddings(in_indices)
        
        for layer in self.layers:
            x = layer(x, token_positions)
        
        x = self.ln_final(x)
        logits = self.lm_head(x)
        
        return logits
