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

# These settings seem to work well

generator = ExLlamaGenerator(model, tokenizer, cache)
generator.settings = ExLlamaGenerator.Settings()
generator.settings.top_k = 20
generator.settings.top_p = 0.65
generator.settings.min_p = 0.02
generator.settings.token_repetition_penalty_max = 1.2
generator.settings.token_repetition_penalty_sustain = 50
generator.settings.token_repetition_penalty_decay = 50

# Be nice to Chatbort

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

    # Read and format input

    in_line = input(username + ": ")
    in_line = username + ": " + in_line.strip() + "\n"

    # No need for this, really

    past += in_line

    # SentencePiece doesn't tokenize spaces separately so we can't know from individual tokens if they start a new word
    # or not. Instead, repeatedly decode the generated response as it's being built, starting from the last newline,
    # and print out the differences between consecutive decodings to stream out the response.

    in_tokens = tokenizer.encode(in_line)

    res_line = bot_name + ":"
    res_tokens = tokenizer.encode(res_line)
    num_res_tokens = res_tokens.shape[-1]  # Decode from here

    in_tokens = torch.cat((in_tokens, res_tokens), dim = 1)

    # If we're approaching the context limit, prune some whole lines from the start of the context. Also prune a
    # little extra so we don't end up rebuilding the cache on ever line when up against the limit.

    expect_tokens = in_tokens.shape[-1] + max_response_tokens
    max_tokens = config.max_seq_len - expect_tokens
    if generator.gen_num_tokens() >= max_tokens:
        generator.gen_prune_to(config.max_seq_len - expect_tokens - extra_prune, tokenizer.newline_token_id)

    # Feed in the user input and "{bot_name}:", tokenized

    generator.gen_feed_tokens(in_tokens)

    # Generate with streaming

    print(res_line, end = "")
    sys.stdout.flush()

    for i in range(max_response_tokens):
        token = generator.gen_single_token()
        generator.gen_accept_token(token)

        num_res_tokens += 1
        text = tokenizer.decode(generator.sequence[:, -num_res_tokens:][0])
        new_text = text[len(res_line):]
        res_line += new_text

        print(new_text, end="")
        sys.stdout.flush()

        if token.item() == tokenizer.newline_token_id: break  # Response includes newline

    past += res_line
