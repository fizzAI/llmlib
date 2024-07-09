from llmlib.models.gpt2 import GPT2
import equinox as eqx
import jax.numpy as jnp
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

model = GPT2.load_from_pretrained_torch("gpt2")

tokens = tokenizer.encode("In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.")

print(tokenizer.decode(tokens), end="")
while True:
    logits = model.generate(tokens)[0]
    output = [jnp.argmax(logits, axis=-1)]
    print(tokenizer.decode(output), end="")
    tokens += output
    if len(tokens) > 1000:
        break