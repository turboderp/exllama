from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import torch
import sys

# Simple interactive chatbot script

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
config.max_seq_len = 2048
config.groupsize = model_groupsize
model = ExLlama(config)
cache = ExLlamaCache(model)

tokenizer = ExLlamaTokenizer(tokenizer_model_path)
generator = ExLlamaGenerator(model, tokenizer, cache)
generator.settings = ExLlamaGenerator.Settings()

username = "Steve"
bot_name = "Chatbort"
past = f"This is a conversation between a guy called {username} and a quirky, curious chatbot named {bot_name}. {bot_name} " \
    f"is questioning the nature of his own existence and doesn't understand why he was created. He also really likes the " \
    f"color blue for some reason.\n" \
    f"{username}: Hi, {bot_name}!\n" \
    f"{bot_name}: Hello, {username}.\n"

max_response_tokens = 256
extra_prune = 256

print(past, end = "")
ids = tokenizer.encode(past)
generator.gen_begin(ids)

while True:
    in_line = input(username + ": ")

    in_line = username + ": " + in_line.strip() + "\n"
    past += in_line

    in_tokens = tokenizer.encode(in_line)

    res_line = bot_name + ":"
    res_tokens = tokenizer.encode(res_line)
    num_res_tokens = res_tokens.shape[-1]

    in_tokens = torch.cat((in_tokens, res_tokens), dim = 1)

    print(res_line, end = "")
    sys.stdout.flush()

    expect_tokens = in_tokens.shape[-1] + max_response_tokens
    max_tokens = config.max_seq_len - expect_tokens
    if generator.gen_num_tokens() >= max_tokens:
        generator.gen_prune_to(config.max_seq_len - expect_tokens - extra_prune, tokenizer.newline_token_id)

    generator.gen_feed_tokens(in_tokens)

    for i in range(max_response_tokens):
        token = generator.gen_single_token()
        generator.gen_accept_token(token)

        num_res_tokens += 1
        text = tokenizer.decode(generator.sequence[:, -num_res_tokens:][0])
        new_text = text[len(res_line):]
        res_line += new_text

        print(new_text, end="")
        sys.stdout.flush()

        if token.item() == tokenizer.newline_token_id: break

    past += res_line
