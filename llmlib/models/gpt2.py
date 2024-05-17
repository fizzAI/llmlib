from jaxtyping import Array, Float, Int
from typing import Optional
import equinox as eqx
import jax.numpy as jnp
import jax

from .template import GenerativeModel
from ..attention.self_attention import SelfAttention
from ..activations import get_activation_by_name
from ..normalization import LayerNorm

class GPT2MultiLayerPerceptron(eqx.Module):
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    actv: eqx.Module

    def __init__(self, dim: Int, intermediate_dim: Int, actv: eqx.Module = "approx_gelu"):
        self.c_fc = eqx.nn.Linear(dim, intermediate_dim)
        self.c_proj = eqx.nn.Linear(intermediate_dim, dim)
        self.actv = get_activation_by_name(actv)

    def __call__(self, x: Array) -> Array:
        return self.c_proj(self.actv(self.c_fc(x)))

class GPT2TransformerBlock(eqx.Module):
    n_heads: Int
    n_kv_heads: Int
    head_dim: Int

    ln1: LayerNorm
    attn: SelfAttention
    ln2: LayerNorm
    mlp: GPT2MultiLayerPerceptron

    def __init__(self, dim: Int, max_seq_len: Int, num_heads: Int = 1, num_kv_heads: Optional[Int] = None):
        self.n_heads = num_heads
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = dim // self.n_heads

        self.ln1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, max_seq_len, num_heads, num_kv_heads)
        self.ln2 = LayerNorm(dim)
        self.mlp = GPT2MultiLayerPerceptron(dim, dim * 4)

    def __call__(self, x: Array) -> Array:
        x = self.ln1(x)
        x = self.attn(x)
        x = self.ln2(x)
        x = self.mlp(x)
        return x

class GPT2(GenerativeModel):
    pass