from model import ExLlama, ExLlamaCache, ExLlamaConfig
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

tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
tokenizer.pad_token_id = 0
tokenizer.bos_token_id = 1
tokenizer.eos_token_id = 2

config = ExLlamaConfig(model_config_path, model_path)
config.attention_method = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP
config.matmul_method = ExLlamaConfig.MatmulMethod.QUANT_ONLY
config.groupsize = model_groupsize
model = ExLlama(config)
cache = ExLlamaCache(model)

gen_tokens = 128
ids = tokenizer.encode("Q: What are five good reasons to sell your house and buy a boat instead?\nA:", return_tensors = "pt", add_special_tokens = False)

with torch.no_grad():

    logits = model.forward(ids, cache)

    while True:

        logits = logits[0, -1, :]
        token = torch.argmax(logits).cpu()
        next_id = token.unsqueeze(0).unsqueeze(0)
        ids = torch.cat((ids, next_id), dim = 1)

        gen_tokens -= 1
        if gen_tokens == 0: break

        logits = model.forward(next_id, cache)

    text = tokenizer.decode(ids[0])
    print(text)



