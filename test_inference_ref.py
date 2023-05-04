from model import ExLlama, ExLlamaCache, ExLlamaConfig
from autograd_ref.autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from transformers import LlamaTokenizer
import torch

# Just a quick test to see if we are getting anything sensible out of the model. Greedy sampling, should produce
# uninteresting and repetitive (but coherent) text

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

model, tokenizer = load_llama_model_4bit_low_ram(tokenizer_path, model_path, groupsize = model_groupsize, is_v1_model = (model_groupsize == -1))

model.half()
for n, m in model.named_modules():
    if isinstance(m, Autograd4bitQuantLinear):
        if m.groupsize == -1: m.zeros = m.zeros.half()
        m.scales = m.scales.half()
        m.bias = m.bias.half()

gen_tokens = 128
ids = tokenizer.encode("Q: What are five good reasons to sell your house and buy a boat instead?\nA:", return_tensors = "pt", add_special_tokens = False)

with torch.no_grad():

    result = model.forward(ids, use_cache = True)
    logits = result["logits"]
    pkv = result["past_key_values"]

    while True:

        logits = logits[0, -1, :]
        token = torch.argmax(logits).cpu()
        next_id = token.unsqueeze(0).unsqueeze(0)
        ids = torch.cat((ids, next_id), dim = 1)

        gen_tokens -= 1
        if gen_tokens == 0: break

        result = model.forward(next_id, past_key_values = pkv, use_cache = True)
        logits = result["logits"]
        pkv = result["past_key_values"]

    text = tokenizer.decode(ids[0])
    print(text)



