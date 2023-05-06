from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
import torch

# Quick test to confirm that caching is working as intended. The two first passes together should produce roughly the
# same logits between them as the third pass, unless causal masking is incorrectly applied for the cached tokens,
# which it seems to be when using the built-in causal modes of SDP and xformers attention. Explicitly supplying a
# correct mask at least works for SDP, although it probably leaves some performance on the table.
# TODO: Make it not be the way that it is but so that it works instead.

tokenizer_model_path = "/mnt/Fast/models/llama-7b-4bit-128g/tokenizer.model"
model_config_path = "/mnt/Fast/models/llama-7b-4bit-128g/config.json"
model_path = "/mnt/Fast/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors"
model_groupsize = 128

config = ExLlamaConfig(model_config_path)
config.model_path = model_path
config.groupsize = 128
config.model_groupsize = model_groupsize
config.attention_method = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP
model = ExLlama(config)
cache = ExLlamaCache(model)

tokenizer = ExLlamaTokenizer(tokenizer_model_path)

ids = tokenizer.encode("Hello!")

with torch.no_grad():

    logits = model.forward(ids, cache, last_id_only = False)
    print(logits)

    logits = model.forward(ids, cache, last_id_only = False)
    print(logits)

    cache.current_seq_len = 0
    ids = torch.cat((ids, ids), dim = 1)
    logits = model.forward(ids, cache, last_id_only = False)
    print(logits)
