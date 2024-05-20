import jax.experimental
from jaxtyping import Array, Float, Int
from typing import Optional
import equinox as eqx
import jax.numpy as jnp
import jax
import json

from .template import GenerativeModel
from ..attention.self_attention import FusedSelfAttention
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
    attn: FusedSelfAttention
    ln2: LayerNorm
    mlp: GPT2MultiLayerPerceptron

    def __init__(self, dim: Int, max_seq_len: Int, num_heads: Int = 1, num_kv_heads: Optional[Int] = None):
        self.n_heads = num_heads
        self.n_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = dim // self.n_heads

        self.ln1 = LayerNorm(dim)
        self.attn = FusedSelfAttention(dim, max_seq_len, num_heads, num_kv_heads)
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
    transformers: list[GPT2TransformerBlock]
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
        self.transformers = [
            GPT2TransformerBlock(dim, max_seq_len, num_heads, num_kv_heads)
            for _ in range(n_layers)
        ]
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
                c_attn_weight = jnp.asarray(f.get_tensor(f"h.{i}.attn.c_attn.weight"))
                c_attn_bias = jnp.asarray(f.get_tensor(f"h.{i}.attn.c_attn.bias")) # shape [2 304] 
                c_attn_layer = eqx.nn.Linear(model.dim, model.dim * 3, key=jax.random.PRNGKey(0))
                c_attn_layer = eqx.tree_at(
                    lambda x: x.weight,
                    c_attn_layer,
                    c_attn_weight
                )
                c_attn_layer = eqx.tree_at(
                    lambda x: x.bias,
                    c_attn_layer,
                    c_attn_bias
                )

                # output proj too (its c_proj for some reason)
                c_proj_weight = jnp.asarray(f.get_tensor(f"h.{i}.attn.c_proj.weight"))
                c_proj_bias = jnp.asarray(f.get_tensor(f"h.{i}.attn.c_proj.bias"))
                c_proj_layer = eqx.nn.Linear(model.dim, model.dim, key=jax.random.PRNGKey(0))
                c_proj_layer = eqx.tree_at(
                    lambda x: x.weight,
                    c_proj_layer,
                    c_proj_weight
                )
                c_proj_layer = eqx.tree_at(
                    lambda x: x.bias,
                    c_proj_layer,
                    c_proj_bias
                )

                new_attn_layer = FusedSelfAttention(model.dim, model.max_seq_len, model.n_heads, model.n_kv_heads)
                new_attn_layer = eqx.tree_at(
                    lambda x: x.qkv_proj,
                    new_attn_layer,
                    c_attn_layer
                )
                new_attn_layer = eqx.tree_at(
                    lambda x: x.o_proj,
                    new_attn_layer,
                    c_proj_layer
                )
                model.transformers[i] = eqx.tree_at(
                    lambda x: x.attn,
                    model.transformers[i],
                    new_attn_layer
                )

                # norms and mlp are a bit simpler
                ln1_weight = jnp.asarray(f.get_tensor(f"h.{i}.ln_1.weight"))
                ln1_bias = jnp.asarray(f.get_tensor(f"h.{i}.ln_1.bias"))
                ln1_layer = LayerNorm(model.dim)
                ln1_layer = eqx.tree_at(
                    lambda x: x.weight,
                    ln1_layer,
                    ln1_weight
                )
                ln1_layer = eqx.tree_at(
                    lambda x: x.bias,
                    ln1_layer,
                    ln1_bias
                )
                model.transformers[i] = eqx.tree_at(
                    lambda x: x.ln1,
                    model.transformers[i],
                    ln1_layer
                )

                ln2_weight = jnp.asarray(f.get_tensor(f"h.{i}.ln_2.weight"))
                ln2_bias = jnp.asarray(f.get_tensor(f"h.{i}.ln_2.bias"))
                ln2_layer = LayerNorm(model.dim)
                ln2_layer = eqx.tree_at(
                    lambda x: x.weight,
                    ln2_layer,
                    ln2_weight
                )
                ln2_layer = eqx.tree_at(
                    lambda x: x.bias,
                    ln2_layer,
                    ln2_bias
                )
                model.transformers[i] = eqx.tree_at(
                    lambda x: x.ln2,
                    model.transformers[i],
                    ln2_layer
                )

                c_fc_weight = jnp.asarray(f.get_tensor(f"h.{i}.mlp.c_fc.weight"))
                c_fc_bias = jnp.asarray(f.get_tensor(f"h.{i}.mlp.c_fc.bias"))
                c_fc_layer = eqx.nn.Linear(model.dim, model.dim * 4, key=jax.random.PRNGKey(0))
                c_fc_layer = eqx.tree_at(
                    lambda x: x.weight,
                    c_fc_layer,
                    c_fc_weight
                )
                c_fc_layer = eqx.tree_at(
                    lambda x: x.bias,
                    c_fc_layer,
                    c_fc_bias
                )

                c_proj_weight = jnp.asarray(f.get_tensor(f"h.{i}.mlp.c_proj.weight"))
                c_proj_bias = jnp.asarray(f.get_tensor(f"h.{i}.mlp.c_proj.bias"))
                c_proj_layer = eqx.nn.Linear(model.dim * 4, model.dim, key=jax.random.PRNGKey(0))
                c_proj_layer = eqx.tree_at(
                    lambda x: x.weight,
                    c_proj_layer,
                    c_proj_weight
                )
                c_proj_layer = eqx.tree_at(
                    lambda x: x.bias,
                    c_proj_layer,
                    c_proj_bias
                )

                new_mlp_layer = GPT2MultiLayerPerceptron(model.dim, model.dim * 4)
                new_mlp_layer = eqx.tree_at(
                    lambda x: x.c_fc,
                    new_mlp_layer,
                    c_fc_layer
                )
                new_mlp_layer = eqx.tree_at(
                    lambda x: x.c_proj,
                    new_mlp_layer,
                    c_proj_layer
                )
                model.transformers[i] = eqx.tree_at(
                    lambda x: x.mlp,
                    model.transformers[i],
                    new_mlp_layer
                )

            # load final layer norm and lm_head
            ln_f_weight = jnp.asarray(f.get_tensor("ln_f.weight"))
            ln_f_bias = jnp.asarray(f.get_tensor("ln_f.bias"))
            ln_f_layer = LayerNorm(model.dim)
            ln_f_layer = eqx.tree_at(
                lambda x: x.weight,
                ln_f_layer,
                ln_f_weight
            )
            ln_f_layer = eqx.tree_at(
                lambda x: x.bias,
                ln_f_layer,
                ln_f_bias
            )
            model.ln_f = ln_f_layer

            model.lm_head = jnp.transpose(wte)

        return model

    def __call__(self, x: Array) -> Array:
        if x.shape[0] > self.max_seq_len:
            raise ValueError(f"Input sequence length {x.shape[0]} exceeds maximum sequence length {self.max_seq_len}")

        pos = jnp.arange(0, x.shape[0])
        
        x = self.wte.weight[x]
        x = x + self.wpe.weight[pos]
        for transformer in self.transformers:
            x = transformer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def generate(self, inp: any, n_output: int = 1):
        inp = jnp.asarray(inp)
        logits = self(inp)
        return logits[-n_output:]