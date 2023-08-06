from model import ExLlama, ExLlamaCache, ExLlamaConfig
from lora import ExLlamaLora
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import argparse
import torch
import sys
import os
import glob
import model_init

# Simple interactive chatbot script

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# Parse arguments

parser = argparse.ArgumentParser(description = "Simple chatbot example for ExLlama")

model_init.add_args(parser)

parser.add_argument("-lora", "--lora", type = str, help = "Path to LoRA binary to use during benchmark")
parser.add_argument("-loracfg", "--lora_config", type = str, help = "Path to LoRA config to use during benchmark")
parser.add_argument("-ld", "--lora_dir", type = str, help = "Path to LoRA config and binary. to use during benchmark")

parser.add_argument("-p", "--prompt", type = str, help = "Prompt file")
parser.add_argument("-un", "--username", type = str, help = "Display name of user", default = "User")
parser.add_argument("-bn", "--botname", type = str, help = "Display name of chatbot", default = "Chatbort")
parser.add_argument("-bf", "--botfirst", action = "store_true", help = "Start chat on bot's turn")

parser.add_argument("-nnl", "--no_newline", action = "store_true", help = "Do not break bot's response on newline (allow multi-paragraph responses)")
parser.add_argument("-temp", "--temperature", type = float, help = "Temperature", default = 0.95)
parser.add_argument("-topk", "--top_k", type = int, help = "Top-K", default = 20)
parser.add_argument("-topp", "--top_p", type = float, help = "Top-P", default = 0.65)
parser.add_argument("-minp", "--min_p", type = float, help = "Min-P", default = 0.00)
parser.add_argument("-repp",  "--repetition_penalty", type = float, help = "Repetition penalty", default = 1.15)
parser.add_argument("-repps", "--repetition_penalty_sustain", type = int, help = "Past length for repetition penalty", default = 256)
parser.add_argument("-beams", "--beams", type = int, help = "Number of beams for beam search", default = 1)
parser.add_argument("-beamlen", "--beam_length", type = int, help = "Number of future tokens to consider", default = 1)

args = parser.parse_args()
model_init.post_parse(args)
model_init.get_model_files(args)

# Paths

if args.lora_dir is not None:
    args.lora_config = os.path.join(args.lora_dir, "adapter_config.json")
    args.lora = os.path.join(args.lora_dir, "adapter_model.bin")

# Some feedback

print(f" -- Sequence length: {args.length}")
print(f" -- Temperature: {args.temperature:.2f}")
print(f" -- Top-K: {args.top_k}")
print(f" -- Top-P: {args.top_p:.2f}")
print(f" -- Min-P: {args.min_p:.2f}")
print(f" -- Repetition penalty: {args.repetition_penalty:.2f}")
print(f" -- Beams: {args.beams} x {args.beam_length}")

print_opts = []
if args.no_newline: print_opts.append("no_newline")
if args.botfirst: print_opts.append("botfirst")

model_init.print_options(args, print_opts)

# Globals

model_init.set_globals(args)

# Load prompt file

username = args.username
bot_name = args.botname

if args.prompt is not None:
    with open(args.prompt, "r") as f:
        past = f.read()
        past = past.replace("{username}", username)
        past = past.replace("{bot_name}", bot_name)
        past = past.strip() + "\n"
else:
    past = f"{bot_name}: Hello, {username}\n"

# past += "User: Hi. Please say \"Shhhhhh\"?\n"
# args.botfirst = True

# Instantiate model and generator

config = model_init.make_config(args)

model = ExLlama(config)
cache = ExLlamaCache(model)
tokenizer = ExLlamaTokenizer(args.tokenizer)

model_init.print_stats(model)

# Load LoRA

lora = None
if args.lora:
    print(f" -- LoRA config: {args.lora_config}")
    print(f" -- Loading LoRA: {args.lora}")
    if args.lora_config is None:
        print(f" ## Error: please specify lora path to adapter_config.json")
        sys.exit()
    lora = ExLlamaLora(model, args.lora_config, args.lora)
    if lora.bias_ignored:
        print(f" !! Warning: LoRA zero bias ignored")

# Generator

generator = ExLlamaGenerator(model, tokenizer, cache)
generator.settings = ExLlamaGenerator.Settings()
generator.settings.temperature = args.temperature
generator.settings.top_k = args.top_k
generator.settings.top_p = args.top_p
generator.settings.min_p = args.min_p
generator.settings.token_repetition_penalty_max = args.repetition_penalty
generator.settings.token_repetition_penalty_sustain = args.repetition_penalty_sustain
generator.settings.token_repetition_penalty_decay = generator.settings.token_repetition_penalty_sustain // 2
generator.settings.beams = args.beams
generator.settings.beam_length = args.beam_length

generator.lora = lora

break_on_newline = not args.no_newline

# Be nice to Chatbort

min_response_tokens = 4
max_response_tokens = 256
extra_prune = 256

print(past, end = "")
ids = tokenizer.encode(past)
generator.gen_begin(ids)

next_userprompt = username + ": "

first_round = True

while True:

    res_line = bot_name + ":"
    res_tokens = tokenizer.encode(res_line)
    num_res_tokens = res_tokens.shape[-1]  # Decode from here

    if first_round and args.botfirst: in_tokens = res_tokens

    else:

        # Read and format input

        in_line = input(next_userprompt)
        in_line = username + ": " + in_line.strip() + "\n"

        next_userprompt = username + ": "

        # No need for this, really, unless we were logging the chat. The actual history we work on is kept in the
        # tokenized sequence in the generator and the state in the cache.

        past += in_line

        # SentencePiece doesn't tokenize spaces separately so we can't know from individual tokens if they start a new word
        # or not. Instead, repeatedly decode the generated response as it's being built, starting from the last newline,
        # and print out the differences between consecutive decodings to stream out the response.

        in_tokens = tokenizer.encode(in_line)
        in_tokens = torch.cat((in_tokens, res_tokens), dim = 1)

    # If we're approaching the context limit, prune some whole lines from the start of the context. Also prune a
    # little extra so we don't end up rebuilding the cache on every line when up against the limit.

    expect_tokens = in_tokens.shape[-1] + max_response_tokens
    max_tokens = config.max_seq_len - expect_tokens
    if generator.gen_num_tokens() >= max_tokens:
        generator.gen_prune_to(config.max_seq_len - expect_tokens - extra_prune, tokenizer.newline_token_id)

    # Feed in the user input and "{bot_name}:", tokenized

    generator.gen_feed_tokens(in_tokens)

    # Generate with streaming

    print(res_line, end = "")
    sys.stdout.flush()

    generator.begin_beam_search()

    for i in range(max_response_tokens):

        # Disallowing the end condition tokens seems like a clean way to force longer replies.

        if i < min_response_tokens:
            generator.disallow_tokens([tokenizer.newline_token_id, tokenizer.eos_token_id])
        else:
            generator.disallow_tokens(None)

        # Get a token

        gen_token = generator.beam_search()

        # If token is EOS, replace it with newline before continuing

        if gen_token.item() == tokenizer.eos_token_id:
            generator.replace_last_token(tokenizer.newline_token_id)

        # Decode the current line and print any characters added

        num_res_tokens += 1
        text = tokenizer.decode(generator.sequence_actual[:, -num_res_tokens:][0])
        new_text = text[len(res_line):]

        skip_space = res_line.endswith("\n") and new_text.startswith(" ")  # Bit prettier console output
        res_line += new_text
        if skip_space: new_text = new_text[1:]

        print(new_text, end="")  # (character streaming output is here)
        sys.stdout.flush()

        # End conditions

        if break_on_newline and gen_token.item() == tokenizer.newline_token_id: break
        if gen_token.item() == tokenizer.eos_token_id: break

        # Some models will not (or will inconsistently) emit EOS tokens but in a chat sequence will often begin
        # generating for the user instead. Try to catch this and roll back a few tokens to begin the user round.

        if res_line.endswith(f"{username}:"):
            plen = tokenizer.encode(f"{username}:").shape[-1]
            generator.gen_rewind(plen)
            next_userprompt = " "
            break

    generator.end_beam_search()

    past += res_line
    first_round = False
