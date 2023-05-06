from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
import torch

# Just a quick test to see if we are getting anything sensible out of the model. Greedy sampling, should produce
# uninteresting and repetitive (but coherent) text

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# tokenizer_model_path = "/mnt/Fast/models/llama-7b-4bit-128g/tokenizer.model"
# model_config_path = "/mnt/Fast/models/llama-7b-4bit-128g/config.json"
# model_path = "/mnt/Fast/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors"
# model_groupsize = 128
#
# tokenizer_model_path = "/mnt/Fast/models/llama-13b-4bit-128g/tokenizer.model"
# model_config_path = "/mnt/Fast/models/llama-13b-4bit-128g/config.json"
# model_path = "/mnt/Fast/models/llama-13b-4bit-128g/llama-13b-4bit-128g.safetensors"
# model_groupsize = 128

tokenizer_model_path = "/mnt/Fast/models/llama-30b-4bit-128g/tokenizer.model"
model_config_path = "/mnt/Fast/models/llama-30b-4bit-128g/config.json"
model_path = "/mnt/Fast/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors"
model_groupsize = 128

# tokenizer_model_path = "/mnt/Fast/models/llama-30b-4bit-128g-act/tokenizer.model"
# model_config_path = "/mnt/Fast/models/llama-30b-4bit-128g-act/config.json"
# model_path = "/mnt/Fast/models/llama-30b-4bit-128g-act/llama-30b-4bit-128g.safetensors"
# model_groupsize = 128

config = ExLlamaConfig(model_config_path)
config.model_path = model_path
# config.attention_method = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP
# config.matmul_method = ExLlamaConfig.MatmulMethod.QUANT_ONLY
config.max_seq_len = 1536
config.groupsize = model_groupsize
model = ExLlama(config)
cache = ExLlamaCache(model)

tokenizer = ExLlamaTokenizer(tokenizer_model_path)

gen_tokens = 200
ids = tokenizer.encode("Q: What are five good reasons to sell your house and buy a boat instead?\nA:")

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



