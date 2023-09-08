from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import torch
import torch.nn.functional as F
import os, glob
import cuda_ext

# Directory containing model, tokenizer, generator

model_directory =  "/mnt/str/models/_test_models/TheBloke_Llama-2-13B-chat-GPTQ/"

# Locate files we need within that directory

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)

# Create config, model, tokenizer and generator

config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

cache = ExLlamaCache(model, batch_size = 2)             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

# Configure generator

generator.settings.token_repetition_penalty_max = 1.15
generator.settings.temperature = 0.95
generator.settings.top_k = 40
generator.settings.top_p = 0.75
# generator.settings.typical = 0.95

# Prompts to mix

f1 = \
"""[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]"""

f2 = \
"""[INST] <<SYS>>
<</SYS>>
You are a rude and obnoxious assistant. You hate everything and everyone.
{prompt}[/INST]"""


prompts = \
[
    f1.replace("{prompt}", "Tell me about Homer Simpson"),
    f2.replace("{prompt}", "Tell me about Homer Simpson"),
]

def generate_cfg(prompts, alpha, max_new_tokens):

    ids, mask = tokenizer.encode(prompts, return_mask = True)
    generator.gen_begin(ids, mask = mask)

    # Sampling loop

    for _ in range(max_new_tokens):

        logits = model.forward(generator.sequence[:, -1:], cache, input_mask = mask)
        generator.apply_rep_penalty(logits)

        logits = F.log_softmax(logits, dim = -1)
        logits_mixed = (1 - alpha) * logits[0] + alpha * logits[1]

        sampled_token, _ = generator.sample_current(logits_mixed)
        if sampled_token.item() == tokenizer.eos_token_id: break

        batch_token = sampled_token.repeat(2, 1)
        generator.gen_accept_token(batch_token)

    output = tokenizer.decode(generator.sequence[0])
    return output

for i in range(10):

    alpha = i / 5.0 - 0.4
    print()
    print(f"--------------------------------------")
    print(f"alpha = {alpha:.1f}")
    print(f"--------------------------------------")
    output = generate_cfg(prompts, alpha, 200)
    print(output[len(prompts[0]):].strip())
