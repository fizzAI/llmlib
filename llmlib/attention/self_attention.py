from jaxtyping import Array, Float, Int
from typing import Optional
import equinox as eqx
import jax.numpy as jnp

# basic self-attention module () as described in Attention is All You Need
class SelfAttention(eqx.Module):
    n_heads: Int
    n_kv_heads: Int
    n_rep: Int
    head_dim: Int

    wq: Array
    wk: Array
    wv: Array
    wo: Array

    def __init__(self, dim: Int, num_heads: Int = 1, num_kv_heads: Optional[Int] = None):
        self.n_heads = num_heads
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // self.n_heads

        self.wq = eqx.nn.Linear(dim, self.n_heads * self.head_dim, use_bias=False)
        self.wk = eqx.nn.Linear(dim, self.n_kv_heads * self.head_dim, use_bias=False)
        self.wv = eqx.nn.Linear(dim, self.n_kv_heads * self.head_dim, use_bias=False)
        self.wo = eqx.nn.Linear(self.n_heads * self.head_dim, dim, use_bias=False)

        # FIXME: flash-attn impl? (Tri Dao et al. 2021)
        pass

    def __call__(self, x: Array, freqs_cos: Array, freqs_sin: Array) -> Array:
        pass

class SelfAttentionWithRoPE(SelfAttention):
    def __call__(self, x: Array, freqs_cos: Array, freqs_sin: Array) -> Array:
        pass