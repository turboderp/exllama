import torch

import cProfile, pstats, io
from pstats import SortKey

from .model import ExLlama, ExLlamaCache, ExLlamaConfig
from .tokenizer import ExLlamaTokenizer

tokenizer_model_path = "/mnt/str/models/llama-30b-4bit-128g/tokenizer.model"
model_config_path = "/mnt/str/models/llama-30b-4bit-128g/config.json"
model_path = "/mnt/str/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors"

tokenizer = ExLlamaTokenizer(tokenizer_model_path)

config = ExLlamaConfig(model_config_path)
config.model_path = model_path
model = ExLlama(config)
cache = ExLlamaCache(model)

ids = torch.randint(0, 31999, (1, 1024))

pr = cProfile.Profile()
pr.enable()

with torch.no_grad():
    for i in range(128):
        model.forward(ids, cache)
        ids = torch.randint(0, 31999, (1, 1))
    cache.current_seq_len = 0

pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())
