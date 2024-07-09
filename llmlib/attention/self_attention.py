from jaxtyping import Array, Float, Int
from typing import Optional
import equinox as eqx
import jax.numpy as jnp
import jax

# basic self-attention module () as described in Attention is All You Need
class FusedSelfAttention(eqx.Module):
    dim: Int
    n_heads: Int
    n_kv_heads: Int
    head_dim: Int

    qkv_proj: eqx.nn.Linear
    o_proj: eqx.nn.Linear

    mask: Array

    def __init__(self, dim: int, max_seq_len: int, num_heads: int = 1, num_kv_heads: Optional[int] = None):
        self.dim = dim
        self.n_heads = num_heads
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = dim // num_heads
        
        self.qkv_proj = eqx.nn.Linear(dim, 3 * self.n_heads * self.head_dim, use_bias=False, key=jax.random.PRNGKey(0))
        self.o_proj = eqx.nn.Linear(self.n_heads * self.head_dim, dim, use_bias=False, key=jax.random.PRNGKey(0))

        self.mask = jnp.tril(jnp.ones((max_seq_len, max_seq_len)))

    def __call__(self, x: jax.Array) -> jax.Array:
        # x.shape == (batch_size, seq_len, dim)
        T, C = x.shape
        assert C == self.dim
        
        qkv = x @ self.qkv_proj.weight
        q, k, v = jnp.split(qkv, [self.n_heads * self.head_dim, (self.n_heads + self.n_kv_heads) * self.head_dim], axis=-1)
        q = q.reshape(T, self.n_heads, self.head_dim).transpose(1, 0, 2)
        k = k.reshape(T, self.n_kv_heads, self.head_dim).transpose(1, 0, 2)
        v = v.reshape(T, self.n_kv_heads, self.head_dim).transpose(1, 0, 2)

        # If n_kv_heads < n_heads, we need to repeat k and v
        if self.n_kv_heads < self.n_heads:
            k = jnp.repeat(k, self.n_heads // self.n_kv_heads, axis=1)
            v = jnp.repeat(v, self.n_heads // self.n_kv_heads, axis=1)

        scores = jnp.einsum('hid,hjd->hij', q, k) / jnp.sqrt(self.head_dim)
        mask = self.mask[None, :T, :T]
        scores = jnp.where(mask, scores, -jnp.inf)

        attn = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('hij,hjd->hid', attn, v)
        out = out.transpose(1, 0, 2).reshape(T, self.dim)
        out = out @ self.o_proj.weight

        return out

""" # TODO

class SelfAttentionWithRoPE(SelfAttention):
    def __call__(self, x: Array, freqs_cos: Array, freqs_sin: Array) -> Array:
        pass """