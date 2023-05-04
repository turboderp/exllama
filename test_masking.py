
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from transformers import LlamaTokenizer
import time
import torch

# Quick test to confirm that caching is working as intended. The two first passes together should produce roughly the
# same logits between them as the third pass, unless causal masking is incorrectly applied for the cached tokens,
# which it seems to be when using the built-in causal modes of SDP and xformers attention. Explicitly supplying a
# correct mask at least works for SDP, although it probably leaves some performance on the table.
# TODO: Make it not be the way that it is but so that it works instead.

tokenizer_path = "/mnt/Fast/models/llama-7b-4bit-128g/"
model_config_path = "/mnt/Fast/models/llama-7b-4bit-128g/config.json"
model_path = "/mnt/Fast/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors"
model_groupsize = 128

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

config = ExLlamaConfig(model_config_path, model_path)
model = ExLlama(config, model_groupsize)
cache = ExLlamaCache(model)

gen_tokens = 128
ids = tokenizer.encode("Hello!", return_tensors = "pt")

with torch.no_grad():

    logits = model.forward(ids, cache)
    print(logits)

    logits = model.forward(ids, cache)
    print(logits)

    cache.current_seq_len = 0
    ids = torch.cat((ids, ids), dim = 1)
    logits = model.forward(ids, cache)
    print(logits)
