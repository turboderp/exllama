
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from transformers import LlamaTokenizer
import torch

import cProfile, pstats, io
from pstats import SortKey

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
ids = torch.randint(0, 31999, (1, config.max_seq_len - gen_tokens))

pr = cProfile.Profile()
pr.enable()

with torch.no_grad():
    model.forward(ids, cache)
    cache.current_seq_len = 0
    model.forward(ids, cache)
    cache.current_seq_len = 0
    model.forward(ids, cache)
    cache.current_seq_len = 0
    model.forward(ids, cache)

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
