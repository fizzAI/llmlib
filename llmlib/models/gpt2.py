import jax.experimental
from jaxtyping import Array, Float, Int
from typing import Optional
import equinox as eqx
import jax.numpy as jnp
import jax
import json

from .template import GenerativeModel
from ..attention.self_attention import SelfAttention
from ..activations import get_activation_by_name
from ..normalization import LayerNorm

class GPT2MultiLayerPerceptron(eqx.Module):
    c_fc: eqx.nn.Linear
    c_proj: eqx.nn.Linear

    actv: eqx.Module

    def __init__(self, dim: Int, intermediate_dim: Int, actv: eqx.Module = "approx_gelu"):
        self.c_fc = eqx.nn.Linear(dim, intermediate_dim, key=jax.random.PRNGKey(0))
        self.c_proj = eqx.nn.Linear(intermediate_dim, dim, key=jax.random.PRNGKey(0))
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
    n_layers: Int
    dim: Int
    max_seq_len: Int
    n_heads: Int
    n_kv_heads: Int

    wte: eqx.nn.Embedding
    wpe: eqx.nn.Embedding
    transformers: eqx.nn.Sequential
    ln_f: LayerNorm
    lm_head: eqx.nn.Linear

    def __init__(self, vocab_size: Int, dim: Int, max_seq_len: Int, n_layers: Int, num_heads: Int = 1, num_kv_heads: Optional[Int] = None):
        self.n_layers = n_layers
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_heads = num_heads
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads

        self.wte = eqx.nn.Embedding(vocab_size, dim, key=jax.random.PRNGKey(0))
        self.wpe = eqx.nn.Embedding(max_seq_len, dim, key=jax.random.PRNGKey(0)) # i think??
        self.transformers = eqx.nn.Sequential([
            GPT2TransformerBlock(dim, max_seq_len, num_heads, num_kv_heads)
            for _ in range(n_layers)
        ])
        self.ln_f = LayerNorm(dim)
        self.lm_head = eqx.nn.Linear(dim, vocab_size, use_bias=False, key=jax.random.PRNGKey(0))

    @staticmethod
    def load_from_pretrained_torch(path: str):
        from safetensors import safe_open
        # FIXME: download directly from HuggingFace?

        # initialize model with config
        with open(f"{path}/tokenizer.json", "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)
        model = GPT2(
            vocab_size=len(tokenizer_config["model"]["vocab"]) ,
            dim=config["n_embd"],
            max_seq_len=config["n_ctx"],
            n_layers=config["n_layer"],
            num_heads=config["n_head"],
            num_kv_heads=config["n_head"]
        )

        # load weights from safetensors
        with safe_open(f"{path}/model.safetensors", framework="np") as f:
            wpe = jnp.asarray(f.get_tensor("wpe.weight"))
            model.wpe = eqx.tree_at(
                lambda x: x.weight,
                model.wpe,
                wpe
            )
            wte = jnp.asarray(f.get_tensor("wte.weight"))
            model.wte = eqx.tree_at(
                lambda x: x.weight,
                model.wte,
                wte
            )

            for i, transformer in enumerate(model.transformers):
                # their attention qkv linears are concatenated into one `c_attn` tensor. we need to split them
                c_attn = jnp.asarray(f.get_tensor(f"h.{i}.attn.c_attn.weight"))
                c_attn = c_attn.reshape(3, -1, model.dim)
                model.transformers.layers[0].attn.q_proj = eqx.tree_at(
                    lambda x: x.weight,
                    transformer.attn.q_proj,
                    c_attn[0]
                )
                model.transformers.layers[0].attn.k_proj = eqx.tree_at(
                    lambda x: x.weight,
                    transformer.attn.k_proj,
                    c_attn[1]
                )
                model.transformers.layers[0].attn.v_proj = eqx.tree_at(
                    lambda x: x.weight,
                    transformer.attn.v_proj,
                    c_attn[2]
                )
                # load other attn weights. output projection is in c_proj for some reason
                model.transformers.layers[0].attn.o_proj = eqx.tree_at(
                    lambda x: x.weight,
                    transformer.attn.o_proj,
                    jnp.asarray(f.get_tensor(f"h.{i}.attn.c_proj.weight"))
                )

                # same but for biases
                c_attn = jnp.asarray(f.get_tensor(f"h.{i}.attn.c_attn.bias"))
                c_attn = c_attn.reshape(3, -1)
                model.transformers.layers[0].attn.q_proj = eqx.tree_at(
                    lambda x: x.bias,
                    transformer.attn.q_proj,
                    c_attn[0]
                )
                model.transformers.layers[0].attn.k_proj = eqx.tree_at(
                    lambda x: x.bias,
                    transformer.attn.k_proj,
                    c_attn[1]
                )
                model.transformers.layers[0].attn.v_proj = eqx.tree_at(
                    lambda x: x.bias,
                    transformer.attn.v_proj,
                    c_attn[2]
                )
                # load other attn biases
                model.transformers.layers[0].attn.o_proj = eqx.tree_at(
                    lambda x: x.bias,
                    transformer.attn.o_proj,
                    jnp.asarray(f.get_tensor(f"h.{i}.attn.c_proj.bias"))
                )
                # FIXME: there's apparently a generic bias here? never seen it in the implementations
                
                # load the rest of the weights
                model.transformers.layers[0].mlp.c_fc = eqx.tree_at(
                    lambda x: x.weight,
                    transformer.mlp.c_fc,
                    jnp.asarray(f.get_tensor(f"h.{i}.mlp.c_fc.weight"))
                )
                model.transformers.layers[0].mlp.c_fc = eqx.tree_at(
                    lambda x: x.bias,
                    transformer.mlp.c_fc,
                    jnp.asarray(f.get_tensor(f"h.{i}.mlp.c_fc.bias"))
                )
                model.transformers.layers[0].mlp.c_proj = eqx.tree_at(
                    lambda x: x.weight,
                    transformer.mlp.c_proj,
                    jnp.asarray(f.get_tensor(f"h.{i}.mlp.c_proj.weight"))
                )
                model.transformers.layers[0].mlp.c_proj = eqx.tree_at(
                    lambda x: x.bias,
                    transformer.mlp.c_proj,
                    jnp.asarray(f.get_tensor(f"h.{i}.mlp.c_proj.bias"))
                )

                model.transformers.layers[0].ln1 = eqx.tree_at(
                    lambda x: x.weight,
                    transformer.ln1,
                    jnp.asarray(f.get_tensor(f"h.{i}.ln_1.weight"))
                )
                model.transformers.layers[0].ln1 = eqx.tree_at(
                    lambda x: x.bias,
                    transformer.ln1,
                    jnp.asarray(f.get_tensor(f"h.{i}.ln_1.bias"))
                )
                model.transformers.layers[0].ln2 = eqx.tree_at(
                    lambda x: x.weight,
                    transformer.ln2,
                    jnp.asarray(f.get_tensor(f"h.{i}.ln_2.weight"))
                )
                model.transformers.layers[0].ln2 = eqx.tree_at(
                    lambda x: x.bias,
                    transformer.ln2,
                    jnp.asarray(f.get_tensor(f"h.{i}.ln_2.bias"))
                )


            model.ln_f = eqx.tree_at(
                lambda x: x.weight,
                model.ln_f,
                jnp.asarray(f.get_tensor("ln_f.weight"))
            )
            model.ln_f = eqx.tree_at(
                lambda x: x.bias,
                model.ln_f,
                jnp.asarray(f.get_tensor("ln_f.bias"))
            )

            model.lm_head = jnp.transpose(wte)

    def __call__(self, x: Array) -> Array:
        if x.shape[0] > self.max_seq_len:
            raise ValueError(f"Input sequence length {x.shape[0]} exceeds maximum sequence length {self.max_seq_len}")

        pos = jnp.arange(0, x.shape[0])
        
        x = self.wte(x)
        x = x + self.wpe(pos)
        for transformer in self.transformers:
            x = transformer(x)
        x = self.ln_f(x)
        return self.lm_head(x)