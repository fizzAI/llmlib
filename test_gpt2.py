from llmlib.models.gpt2 import GPT2
import equinox as eqx

model = GPT2.load_from_pretrained_torch("gpt2")
eqx.tree_pprint(model)