from __future__ import annotations

import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    from cs336_basics.layers import Linear
    linear = Linear(d_in, d_out)
    linear.weight.data = weights
    return linear(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.layers import Embedding
    emb = Embedding(vocab_size, d_model)
    emb.weight.data = weights
    return emb(token_ids)


def run_swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.layers import SwiGLUFFN
    swiglu = SwiGLUFFN(d_model, d_ff=d_ff)
    swiglu.w1.weight.data = w1_weight
    swiglu.w2.weight.data = w2_weight
    swiglu.w3.weight.data = w3_weight
    return swiglu(in_features)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    from cs336_basics.attention_stack import scaled_dot_product_attention
    return scaled_dot_product_attention(Q, K, V, mask=mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    from cs336_basics.attention_stack import MultiHeadSelfAttentionNoRoPE
    mha = MultiHeadSelfAttentionNoRoPE(d_model=d_model, n_heads=num_heads)
    mha.w_q.weight.data = q_proj_weight
    mha.w_k.weight.data = k_proj_weight
    mha.w_v.weight.data = v_proj_weight
    mha.w_o.weight.data = o_proj_weight
    return mha(in_features)


def run_multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_out"]:
    from cs336_basics.attention_stack import MultiHeadSelfAttention
    mha = MultiHeadSelfAttention(
        d_model=d_model,
        n_heads=num_heads,
        rope_theta=theta,
        max_seq_len=max_seq_len,
    )
    mha.w_q.weight.data = q_proj_weight
    mha.w_k.weight.data = k_proj_weight
    mha.w_v.weight.data = v_proj_weight
    mha.w_o.weight.data = o_proj_weight
    
    if token_positions is None:
        seq_len = in_features.shape[-2]
        token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0)
    
    return mha(in_features, token_positions)


def run_rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    from cs336_basics.attention_stack import RotaryPositionalEmbedding
    rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=in_query_or_key.device)
    return rope(in_query_or_key, token_positions)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    from cs336_basics.transformer_block import TransformerBlock
    
    block = TransformerBlock(
        d_model=d_model,
        n_heads=num_heads,
        d_ff=d_ff,
        rope_theta=theta,
        max_seq_len=max_seq_len,
    )
    
    block.norm1.weight.data = weights["ln1.weight"]
    block.attn.w_q.weight.data = weights["attn.q_proj.weight"]
    block.attn.w_k.weight.data = weights["attn.k_proj.weight"]
    block.attn.w_v.weight.data = weights["attn.v_proj.weight"]
    block.attn.w_o.weight.data = weights["attn.output_proj.weight"]
    block.norm2.weight.data = weights["ln2.weight"]
    block.ffn.w1.weight.data = weights["ffn.w1.weight"]
    block.ffn.w2.weight.data = weights["ffn.w2.weight"]
    block.ffn.w3.weight.data = weights["ffn.w3.weight"]
    
    batch_size, seq_len, _ = in_features.shape
    token_positions = torch.arange(seq_len, device=in_features.device).unsqueeze(0).expand(batch_size, -1)
    
    return block(in_features, token_positions)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    from cs336_basics.transformer_block import TransformerLM
    
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
    )
    
    model.token_embeddings.weight.data = weights["token_embeddings.weight"]
    model.ln_final.weight.data = weights["ln_final.weight"]
    model.lm_head.weight.data = weights["lm_head.weight"]
    
    for i in range(num_layers):
        model.layers[i].norm1.weight.data = weights[f"layers.{i}.ln1.weight"]
        model.layers[i].attn.w_q.weight.data = weights[f"layers.{i}.attn.q_proj.weight"]
        model.layers[i].attn.w_k.weight.data = weights[f"layers.{i}.attn.k_proj.weight"]
        model.layers[i].attn.w_v.weight.data = weights[f"layers.{i}.attn.v_proj.weight"]
        model.layers[i].attn.w_o.weight.data = weights[f"layers.{i}.attn.output_proj.weight"]
        model.layers[i].norm2.weight.data = weights[f"layers.{i}.ln2.weight"]
        model.layers[i].ffn.w1.weight.data = weights[f"layers.{i}.ffn.w1.weight"]
        model.layers[i].ffn.w2.weight.data = weights[f"layers.{i}.ffn.w2.weight"]
        model.layers[i].ffn.w3.weight.data = weights[f"layers.{i}.ffn.w3.weight"]
    
    return model(in_indices)


def run_rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    from cs336_basics.layers import RMSNorm
    norm = RMSNorm(d_model, eps=eps)
    norm.weight.data = weights
    return norm(in_features)


def run_silu(in_features: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    return torch.nn.functional.silu(in_features)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    from cs336_basics.train_utils import get_batch
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    from cs336_basics.attention_stack import softmax
    return softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    from cs336_basics.train_utils import cross_entropy
    return cross_entropy(inputs, targets)


def run_gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    from cs336_basics.train_utils import gradient_clipping
    gradient_clipping(parameters, max_l2_norm)


def get_adamw_cls() -> Any:
    from cs336_basics.train_utils import AdamW
    return AdamW


def run_get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    from cs336_basics.train_utils import get_lr_cosine_schedule
    return get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters)


def run_save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    from cs336_basics.train_utils import save_checkpoint
    save_checkpoint(model, optimizer, iteration, out)


def run_load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    from cs336_basics.train_utils import load_checkpoint
    return load_checkpoint(src, model, optimizer)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    """Given a vocabulary, a list of merges, and a list of special tokens,
    return a BPE tokenizer that uses the provided vocab, merges, and special tokens.

    Args:
        vocab (dict[int, bytes]): The tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
            to bytes (token bytes)
        merges (list[tuple[bytes, bytes]]): BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
            representing that <token1> was merged with <token2>.
            Merges are ordered by order of creation.
        special_tokens (list[str] | None): A list of string special tokens for the tokenizer. These strings will never
            be split into multiple tokens, and will always be kept as a single token.

    Returns:
        A BPE tokenizer that uses the provided vocab, merges, and special tokens.
    """
    from cs336_basics.tokenizer import Tokenizer
    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens or [])


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """
    from cs336_basics.bpe_train import train_bpe
    return train_bpe(input_path=str(input_path), vocab_size=vocab_size, special_tokens=special_tokens)
