
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from transformers import LlamaTokenizer
import time
import torch

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# tokenizer_path = "/mnt/Fast/models/llama-7b-4bit-128g/"
# model_config_path = "/mnt/Fast/models/llama-7b-4bit-128g/config.json"
# model_path = "/mnt/Fast/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors"
# model_groupsize = 128
#
# tokenizer_path = "/mnt/Fast/models/llama-13b-4bit-128g/"
# model_config_path = "/mnt/Fast/models/llama-13b-4bit-128g/config.json"
# model_path = "/mnt/Fast/models/llama-13b-4bit-128g/llama-13b-4bit-128g.safetensors"
# model_groupsize = 128

tokenizer_path = "/mnt/Fast/models/llama-30b-4bit-128g/"
model_config_path = "/mnt/Fast/models/llama-30b-4bit-128g/config.json"
model_path = "/mnt/Fast/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors"
model_groupsize = 128

def timer(name, func):
    t = time.time()
    ret = func()
    t = time.time() - t
    print(f" ** Time, {name}: {t:.2f} seconds")
    return ret

torch.cuda.reset_peak_memory_stats("cuda")

mem_base = torch.cuda.max_memory_allocated("cuda")
mem_last = mem_base

def mem(name, total = False):
    global mem_base, mem_last
    mem_c = torch.cuda.max_memory_allocated("cuda")
    mem_this = mem_c - mem_last if not total else mem_c - mem_base
    mem_last = mem_c
    print(f" ** VRAM, {name}: {mem_this / (1024 ** 2):,.2f} MB")

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

config = ExLlamaConfig(model_config_path, model_path)
config.max_seq_len = 2048
config.attention_method = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP
config.matmul_method = ExLlamaConfig.MatmulMethod.QUANT_ONLY
gen_tokens = 128

ids = torch.randint(0, 31999, (1, config.max_seq_len - gen_tokens))

model = timer("load model", lambda: ExLlama(config, model_groupsize))
torch.cuda.reset_peak_memory_stats("cuda")
mem("model")

with torch.no_grad():

    cache = ExLlamaCache(model)
    mem("cache")

    t = time.time()

    print(" -- Inference, first pass.")
    logits = timer("inference", lambda: model.forward(ids, cache))

    t = time.time() - t
    print(f" ** Speed: {ids.shape[-1] / t:.2f} tokens/second")

    t = time.time()

    print(f" -- Generating {gen_tokens} tokens...")
    for i in range(gen_tokens):

        logits = logits[0, -1, :]
        token = torch.argmax(logits)

        next_ids = token.unsqueeze(0).unsqueeze(0)
        logits = model.forward(next_ids, cache)

    t = time.time() - t
    print(f" ** Speed: {gen_tokens / t:.2f} tokens/second")

    mem("inference")
    mem("total", total = True)




