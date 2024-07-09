# Hendrycks et. al 2016

from jaxtyping import Array, Float, Int
from typing import Optional
import equinox as eqx
import jax.numpy as jnp
import jax

class GELU(eqx.Module):
    @jax.jit
    def __call__(self, x: Array):
        return (
            x / 2
        ) * (
            1 + jax.lax.erf(
                x / jnp.sqrt(2)
            )
        )

class ApproximateGELU(eqx.Module):
    @jax.jit
    def __call__(self, x: Array):
        return x * (
            1 + jnp.tanh(
                jnp.sqrt(
                    2 / jnp.pi
                ) * (
                    x + 0.44715 * jnp.power(x, 3)
                )
            )
        ) * 0.5