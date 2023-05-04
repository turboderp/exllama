
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from transformers import LlamaTokenizer
from autograd_ref.autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
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

seq_len = 1024
gen_tokens = 128

model, tokenizer = timer("load model", lambda: load_llama_model_4bit_low_ram(tokenizer_path, model_path, groupsize = model_groupsize, is_v1_model = (model_groupsize == -1)))

model.half()
for n, m in model.named_modules():
    if isinstance(m, Autograd4bitQuantLinear):
        if m.groupsize == -1: m.zeros = m.zeros.half()
        m.scales = m.scales.half()
        m.bias = m.bias.half()

torch.cuda.reset_peak_memory_stats("cuda")
mem("model")

ids = torch.randint(0, 31999, (1, seq_len - gen_tokens)).cuda()

with torch.no_grad():

    t = time.time()

    print(" -- Inference, first pass.")
    result = timer("inference", lambda: model.forward(ids, use_cache = True))
    logits = result["logits"]
    pkv = result["past_key_values"]

    t = time.time() - t
    print(f" ** Speed: {ids.shape[-1] / t:.2f} tokens/second")

    t = time.time()

    print(f" -- Generating {gen_tokens} tokens...")
    for i in range(gen_tokens):

        logits = logits[0, -1, :]
        token = torch.argmax(logits)

        next_ids = token.unsqueeze(0).unsqueeze(0).cuda()
        result = model.forward(next_ids, past_key_values = pkv,  use_cache = True)
        logits = result["logits"]
        pkv = result["past_key_values"]

    t = time.time() - t
    print(f" ** Speed: {gen_tokens / t:.2f} tokens/second")

    mem("inference")
    mem("total", total = True)




