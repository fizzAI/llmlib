from jaxtyping import Array, Float, Int
from typing import Optional
import equinox as eqx
import jax.numpy as jnp
import jax

# basic self-attention module () as described in Attention is All You Need
class SelfAttention(eqx.Module):
    n_heads: Int
    n_kv_heads: Int
    n_rep: Int
    head_dim: Int

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear

    mask: Array

    def __init__(self, dim: Int, max_seq_len: Int, num_heads: Int = 1, num_kv_heads: Optional[Int] = None):
        self.n_heads = num_heads
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = dim // self.n_heads

        self.q_proj = eqx.nn.Linear(dim, self.n_heads * self.head_dim, use_bias=False)
        self.k_proj = eqx.nn.Linear(dim, self.n_kv_heads * self.head_dim, use_bias=False)
        self.v_proj = eqx.nn.Linear(dim, self.n_kv_heads * self.head_dim, use_bias=False)
        self.o_proj = eqx.nn.Linear(self.n_heads * self.head_dim, dim, use_bias=False)

        self.mask = jnp.full((1, 1, max_seq_len, max_seq_len), -jnp.inf)
        self.mask = jnp.triu(self.mask, k=1)

        # FIXME: flash-attn impl? (Tri Dao et al. 2021)
        pass

    def __call__(self, x: Array) -> Array:
        seqlen, _ = x.shape()

        # QKV
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = jnp.transpose(xq.view(seqlen, self.n_heads, self.head_dim), (1, 2))
        xk = jnp.transpose(xk.view(seqlen, self.n_kv_heads, self.head_dim), (1, 2))
        xv = jnp.transpose(xv.view(seqlen, self.n_kv_heads, self.head_dim), (1, 2))

        # dot-product attention
        scores = jnp.matmul(xq, jnp.transpose(xk, (2, 3))) / jax.lax.sqrt(self.head_dim)
        scores = scores + self.mask[:, :, :seqlen, :seqlen] # FIXME: do we need this for causal attention?
        scores = jax.nn.softmax(scores, axis=-1)

        xo = jnp.matmul(scores, xv)

        xo = jnp.transpose(xo, (1, 2)).view(seqlen, -1)

        return self.o_proj(xo)

class SelfAttentionWithRoPE(SelfAttention):
    def __call__(self, x: Array, freqs_cos: Array, freqs_sin: Array) -> Array:
        pass