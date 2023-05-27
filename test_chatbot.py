from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import argparse
import torch
import sys
import os
import glob

# Simple interactive chatbot script

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# Parse arguments

parser = argparse.ArgumentParser(description = "Simple chatbot example for ExLlama")

parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path")
parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)")
parser.add_argument("-m", "--model", type = str, help = "Model weights path (.pt or .safetensors file)")
parser.add_argument("-d", "--directory", type = str, help = "Path to directory containing config.json, model.tokenizer and * .safetensors")

parser.add_argument("-a", "--attention", type = ExLlamaConfig.AttentionMethod.argparse, choices = list(ExLlamaConfig.AttentionMethod), help="Attention method", default = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP)
parser.add_argument("-mm", "--matmul", type = ExLlamaConfig.MatmulMethod.argparse, choices = list(ExLlamaConfig.MatmulMethod), help="Matmul method", default = ExLlamaConfig.MatmulMethod.SWITCHED)
parser.add_argument("-mlp", "--mlp", type = ExLlamaConfig.MLPMethod.argparse, choices = list(ExLlamaConfig.MLPMethod), help="Matmul method", default = ExLlamaConfig.MLPMethod.NORMAL)
parser.add_argument("-s", "--stream", type = int, help = "Stream layer interval", default = 0)
parser.add_argument("-gs", "--gpu_split", type = str, help = "Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7")
parser.add_argument("-dq", "--dequant", type = str, help = "Number of layers (per GPU) to de-quantize at load time")

parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length", default = 2048)

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

parser.add_argument("-gpfix", "--gpu_peer_fix", action = "store_true", help = "Prevent direct copies of data between GPUs")

args = parser.parse_args()

if args.directory is not None:
    args.tokenizer = os.path.join(args.directory, "tokenizer.model")
    args.config = os.path.join(args.directory, "config.json")
    st_pattern = os.path.join(args.directory, "*.safetensors")
    st = glob.glob(st_pattern)
    if len(st) == 0:
        print(f" !! No files matching {st_pattern}")
        sys.exit()
    if len(st) > 1:
        print(f" !! Multiple files matching {st_pattern}")
        sys.exit()
    args.model = st[0]
else:
    if args.tokenizer is None or args.config is None or args.model is None:
        print(" !! Please specify either -d or all of -t, -c and -m")
        sys.exit()

# Some feedback

print(f" -- Loading model")
print(f" -- Tokenizer: {args.tokenizer}")
print(f" -- Model config: {args.config}")
print(f" -- Model: {args.model}")
print(f" -- Sequence length: {args.length}")
print(f" -- Temperature: {args.temperature:.2f}")
print(f" -- Top-K: {args.top_k}")
print(f" -- Top-P: {args.top_p:.2f}")
print(f" -- Min-P: {args.min_p:.2f}")
print(f" -- Repetition penalty: {args.repetition_penalty:.2f}")
print(f" -- Beams: {args.beams} x {args.beam_length}")

print_opts = []
print_opts.append("attention: " + str(args.attention))
print_opts.append("matmul: " + str(args.matmul))
print_opts.append("mlp: " + str(args.mlp))
if args.no_newline: print_opts.append("no_newline")
if args.botfirst: print_opts.append("botfirst")
if args.stream > 0: print_opts.append(f"stream: {args.stream}")
if args.gpu_split is not None: print_opts.append(f"gpu_split: {args.gpu_split}")
if args.dequant is not None: print_opts.append(f"dequant: {args.dequant}")
if args.gpu_peer_fix: print_opts.append("gpu_peer_fix")

print(f" -- Options: {print_opts}")

username = args.username
bot_name = args.botname

# Load prompt file

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

config = ExLlamaConfig(args.config)
config.model_path = args.model
config.attention_method = args.attention
config.matmul_method = args.matmul
config.mlp_method = args.mlp
config.stream_layer_interval = args.stream
config.gpu_peer_fix = args.gpu_peer_fix
if args.length is not None: config.max_seq_len = args.length
config.set_auto_map(args.gpu_split)
config.set_dequant(args.dequant)

model = ExLlama(config)
cache = ExLlamaCache(model)
tokenizer = ExLlamaTokenizer(args.tokenizer)

print(f" -- Groupsize (inferred): {model.config.groupsize if model.config.groupsize is not None else 'None'}")
print(f" -- Act-order (inferred): {'yes' if model.config.act_order else 'no'}")

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
