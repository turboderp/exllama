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


# Llama 2 dialogue
B_INST, E_INST = "[INST]", "[/INST]"            # for user input
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"    # system prompt


# Example for history:
# dialogue = [
#     {"role": "system", "content": "You are a confused assistant."},
#     {"role": "user", "content": "Hi!"},
#     {"role": "assistant", "content": "Hello?"}
# ]






# Llama 2 chat interactive example

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# Parse arguments

parser = argparse.ArgumentParser(description = "Simple chatbot example for ExLlama")

model_init.add_args(parser)

parser.add_argument("-lora", "--lora", type = str, help = "Path to LoRA binary to use during benchmark")
parser.add_argument("-loracfg", "--lora_config", type = str, help = "Path to LoRA config to use during benchmark")
parser.add_argument("-ld", "--lora_dir", type = str, help = "Path to LoRA config and binary. to use during benchmark")

parser.add_argument("-p", "--prompt", type = str, help = "System prompt")
parser.add_argument("-un", "--username", type = str, help = "Display name of user (leave out for standard llama2 chat prompting)", default = "")
parser.add_argument("-bn", "--botname", type = str, help = "Display name of chatbot (leave out for standard llama2 chat prompting)", default = "")

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

model_init.print_options(args, print_opts)


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


# Load prompt file

username = args.username
bot_name = args.botname

if args.prompt is not None:
    with open(args.prompt, "r") as f:
        sysPrompt = f.read()
        sysPrompt = sysPrompt.replace("{username}", username)
        sysPrompt = sysPrompt.replace("{bot_name}", bot_name)
        sysPrompt = sysPrompt.strip() + "\n"
else:
    sysPrompt = f"You are a helpful, respectful and honest assistant{' named ' + bot_name if bot_name else ''}. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.{' You are talking to ' + username + '.' if username else ''}."


dialogue = [
    {"role": "system", "content": sysPrompt}
]


# Main loop

min_response_tokens = 4
max_response_tokens = 1028
extra_prune = 256

print(f"System: {sysPrompt}", end = "\n\n")
ids = tokenizer.encode(sysPrompt, add_bos=True, add_eos=True)
generator.gen_begin(ids)


prompt_tokens = []

while True:
    # Read and format input

    in_line = input(username or "User" + ": ")
    in_line = in_line.strip()         # If you want to include the username in every request, add it in front of the in_line.strip()

    # As the Exllama cache stores the last tokens, we don't really need this line, but it's still helpful if you want to modify past messages etc
    dialogue.append({"role": "user", "content": in_line})

    in_tokens = tokenizer.encode(B_INST + in_line + E_INST, add_bos=True, add_eos=False)    # llama2chat format
    prompt_tokens.append(in_tokens)


    # If we're approaching the context limit, prune some whole lines from the start of the context. Also prune a
    # little extra so we don't end up rebuilding the cache on every line when up against the limit.

    expect_tokens = in_tokens.shape[-1] + max_response_tokens
    max_tokens = config.max_seq_len - expect_tokens
    if generator.gen_num_tokens() >= max_tokens:
        generator.gen_prune_to(config.max_seq_len - expect_tokens - extra_prune, tokenizer.newline_token_id)

    
    generator.gen_feed_tokens(in_tokens)    # Feed in the user input tokenized


    # Generate with streaming

    print(bot_name or "Assistant" + ": ", end = "")
    sys.stdout.flush()

    generator.begin_beam_search()

    full_response = "" # this string gets filled with the streamed strings
    first_non_whitespace = False  # manual trim() of the beginning.
    num_res_tokens = 0    # the number of currently generated tokens. important for the sequence_actual slicing
    for i in range(max_response_tokens):

        # Disallowing the end condition tokens seems like a clean way to force longer replies.

        if i < min_response_tokens:
            generator.disallow_tokens([tokenizer.newline_token_id, tokenizer.eos_token_id])
        else:
            generator.disallow_tokens(None)

        # Get a token

        gen_token = generator.beam_search()

        # Decode the current line and print any characters added
        num_res_tokens += 1
        response = tokenizer.decode(generator.sequence_actual[:, -num_res_tokens:][0])
        new_response = response[len(full_response):]
        
        skip_space = full_response.endswith("\n") and new_response.startswith(" ")  # Bit prettier console output
        full_response += new_response
        if skip_space: new_response = new_response[1:]
        

        print(new_response, end="")  # (character streaming output is here)
        sys.stdout.flush()

        # End condition (EOS is very consistent with Lllama2Chat)
        if gen_token.item() == tokenizer.eos_token_id: break

    generator.end_beam_search()

    print("\n")

    # As said, don't need this specifically but great to have
    dialogue.append({"role": "assistant", "content": full_response})
