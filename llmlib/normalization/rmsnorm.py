from jaxtyping import Array, Float, Int
import equinox as eqx
import jax.numpy as jnp
import jax

# FIXME: apparently the eqx impl supports a bias here? never seen it
# Biao Zhang et. al
class RMSNorm(eqx.Module):
    shape: Int
    scale: Array
    epsilon: Float

    def __init__(self, shape: Int, epsilon: Float = 1e-8):
        self.shape = shape
        self.scale = jnp.ones(shape)
        self.epsilon = epsilon
    
    def __call__(self, x: Array, dtype=jnp.float32) -> Array:
        variance = jnp.asarray(x, dtype=dtype)
        variance = jnp.power(variance, 2).mean(axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)

        return self.scale * x