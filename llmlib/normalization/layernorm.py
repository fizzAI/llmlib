from jaxtyping import Array, Float, Int
from typing import Optional
import equinox as eqx
import jax.numpy as jnp
import jax

class LayerNorm(eqx.Module):
    shape: Int
    epsilon: Float
    weight: Optional[Array]
    bias: Optional[Array]

    def __init__(self, shape: Int, epsilon: Float = 1e-5, use_weight: bool = True, use_bias: bool = True):
        self.shape = shape
        self.weight = jnp.ones(shape) if use_weight else None
        self.bias = jnp.zeros(shape) if use_bias else None
        self.epsilon = epsilon
    
    def __call__(self, x: Array, dtype=jnp.float32) -> Array:
        x = jnp.asarray(x, dtype=dtype)
        mean = jnp.mean(x, keepdims=True)
        variance = jnp.var(x, keepdims=True)
        variance = jnp.maximum(0.0, variance)
        inv = jax.lax.rsqrt(variance + self.epsilon)
        y = (x - mean) * inv
        # FIXME: do i need double negation here
        if not (self.weight is None):
            y = y * self.weight
        if not (self.bias is None):
            y = y + self.bias
        
        return y