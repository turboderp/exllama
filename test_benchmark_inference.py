from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
from lora import ExLlamaLora
import perplexity
from perplexity import Perplexity
import time
import torch
import torch.nn.functional as F
import argparse
import json
import math
import sys
import os
import glob
import model_init

torch.cuda._lazy_init()
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_printoptions(precision = 10)
torch_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

cache = None
model = None

def begin():
    global model, cache

    if cache is None: cache = ExLlamaCache(model)
    else: cache.current_seq_len = 0


def next_logits(input_ids, apply_lora, last_id_only = True, input_mask = None):
    global model, cache

    # n_logits = None
    # a = 0
    # while a < input_ids.shape[-1]:
    #     b = min(input_ids.shape[-1], a + 2048)
    #     n_logits = model.forward(input_ids[:, a:b], cache, last_id_only, lora = apply_lora, input_mask = input_mask)
    #     a = b

    n_logits = model.forward(input_ids, cache, last_id_only, lora=apply_lora, input_mask=input_mask)
    return n_logits


def tokenize(text):
    global tokenizer

    return tokenizer.encode(text)


def timer(name, func):
    t = time.time()
    ret = func()
    t = time.time() - t
    print(f" ** Time, {name}: {t:.2f} seconds")
    return ret


mem_base = {}
mem_last = {}
for dev in torch_devices:
    torch.cuda.reset_peak_memory_stats(dev)
    mem_base[dev] = mem_last[dev] = torch.cuda.max_memory_allocated(dev)

def mem(name, total = False):
    global mem_base, mem_last

    res = f" ** VRAM, {name}: "
    first = True

    for device in torch_devices:
        mem_c = torch.cuda.max_memory_allocated(device)
        mem_this = mem_c - mem_last[device] if not total else mem_c - mem_base[device]
        mem_last[device] = mem_c

        if not first: res += " - "
        first = False
        res += f"[{device}] {mem_this / (1024 ** 2):,.2f} MB"

    print(res)


# Parse arguments

parser = argparse.ArgumentParser(description = "Benchmark tests for ExLlama")

model_init.add_args(parser)
perplexity.add_args(parser)

parser.add_argument("-p", "--perf", action = "store_true", help = "Benchmark speed and VRAM usage")
parser.add_argument("-v", "--validate", action = "count", help = "Run validation check and generate some sample output; specify twice for a more thorough test")
parser.add_argument("-lora", "--lora", type = str, help = "Path to LoRA binary to use during benchmark")
parser.add_argument("-loracfg", "--lora_config", type = str, help = "Path to LoRA config to use during benchmark")
parser.add_argument("-ld", "--lora_dir", type = str, help = "Path to LoRA config and binary. to use during benchmark")

args = parser.parse_args()

model_init.post_parse(args)
perplexity.post_parse(args)
model_init.get_model_files(args)

# Paths

if args.lora_dir is not None:
    args.lora_config = os.path.join(args.lora_dir, "adapter_config.json")
    args.lora = os.path.join(args.lora_dir, "adapter_model.bin")

# Feedback

print_opts = []
if args.perf: print_opts.append("perf")
if args.validate: print_opts.append("validate")
if args.perplexity: print_opts.append("perplexity")
if args.perplexity_token: print_opts.append("perplexity_token")

model_init.print_options(args, print_opts)

# Globals

model_init.set_globals(args)

# Instantiate model

config = model_init.make_config(args)

model = timer("Load model", lambda: ExLlama(config))
tokenizer = timer("Load tokenizer", lambda: ExLlamaTokenizer(args.tokenizer))

model_init.print_stats(model)

torch.cuda.reset_peak_memory_stats("cuda")
mem("Model")

cache = ExLlamaCache(model)
mem("Cache")

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

# Test sequence

gen_tokens = 128
max_seq_len = args.length
ids = torch.randint(0, 31999, (1, max_seq_len - gen_tokens)).cuda()

# Benchmark memory and performance

if args.perf:

    # Warming up apparently makes a huge difference

    for i in range(1, 3):
        print(f" -- Warmup pass {i}...")
        begin()
        logits = timer("Warmup", lambda: next_logits(ids, lora))

    # Do the actual benchmark

    begin()

    t = time.time()

    print(" -- Inference, first pass.")
    logits = timer("Inference", lambda: next_logits(ids, lora))

    t = time.time() - t
    print(f" ** Speed: {ids.shape[-1] / t:.2f} tokens/second")

    for j in range(2):

        t = time.time()
        print(f" -- Generating {gen_tokens} tokens, {ids.shape[-1]} token prompt...")
        for i in range(gen_tokens):

            logits = logits[0, -1, :]
            token = torch.argmax(logits)
            next_id = token.unsqueeze(0).unsqueeze(0)
            logits = next_logits(next_id, lora)

        t = time.time() - t
        print(f" ** Speed: {gen_tokens / t:.2f} tokens/second")

        ids = ids[:, :4]
        cache.current_seq_len = 4

    mem("Inference")
    mem("Total", total = True)


# Benchmark perplexity

if args.perplexity:

    ppl = Perplexity(args.perplexity, model, cache, tokenizer)

    print(" -- Loading dataset...")

    ppl.load(dataset_path = args.perplexity_dataset,
             chunk_size = args.perplexity_chunk_size,
             chunk_truncate = args.perplexity_chunk_truncate,
             overlap = args.perplexity_chunk_overlap,
             minlength = args.perplexity_chunk_min,
             json_key = args.perplexity_json_key)

    begin()

    ppl.test(args.perplexity_chunk_num,
             lora = lora,
             ppl_token = args.perplexity_token)

# Validate file

if args.validate:

    ppl = Perplexity(args.perplexity, model, cache, tokenizer)

    ppl.load(dataset_path = "datasets/wikitext2_val_sample.jsonl",
             chunk_size = 2048,
             chunk_truncate = 2048,
             overlap = 0,
             minlength = 50,
             json_key = "text")

    # Short perplexity tests in switched and quant mode, should produce roughly equal results

    begin()

    ppl.cache.zero()
    model.config.matmul_recons_thd = 1
    ppl.test(8, lora = lora, tag = " (reconstruct)")
    ppl.cache.zero()
    model.config.matmul_recons_thd = 0
    ppl.test(8, lora = lora, tag = " (quant, token)", ppl_token = True)

    # Do a short, easy topk=1 completion to see if we're generating garbage. Should run in switched mode
    # for the prompt and quant for individual tokens

    model.config.matmul_recons_thd = 4
    generator = ExLlamaGenerator(model, tokenizer, cache)
    generator.settings.top_k = 1
    generator.lora = lora
    text = generator.generate_simple("To be or not to be, that is the", max_new_tokens = 20 * args.validate)
    print(f" ** Generation: {repr(text)}")

    if args.validate > 1:

        # Test batched generation

        bsz = 8
        gen_len = 20
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # Bigger cache for the batch

        del cache
        cache = ExLlamaCache(model, batch_size = bsz)

        # Create tokenized batch and attention mask

        identical_batch_prompt = "When you have eliminated the impossible, whatever remains,"
        continuations = [
            " must be considered",
            " ought to be",
            " (and some scholars say this is",
            " however improbable, is a banana.",
        ]

        prompts = [identical_batch_prompt] * (bsz - len(continuations))
        for cont in continuations:
            prompts.append(identical_batch_prompt + cont)

        ids = tokenizer.encode(prompts)
        assert ids.shape[1] < model.config.max_seq_len, f"Max length {ids.shape[1]} exceeds model limit {model.config.max_seq_len}"

        mask = ids.ne(tokenizer.pad_token_id)

        # Batched generation with greedy sampling

        sequence = torch.empty((bsz, 0), dtype = torch.long, device = "cpu")
        logits = next_logits(ids, lora, input_mask = mask)

        for i in range(gen_len):
            logits = logits[:, -1, :]
            id_per_batch = torch.argmax(logits, dim=-1)
            assert id_per_batch.shape == (bsz,), f"{id_per_batch.shape} != {(bsz,)}"
            next_id_per_batch = id_per_batch.unsqueeze(-1)
            sequence = torch.cat((sequence, next_id_per_batch), dim = -1)
            logits = next_logits(next_id_per_batch, lora)

        # Print output batch

        print(f"\n ** Batching sanity check: 1-{bsz - len(continuations)} should be identical. All should be reasonable for the model you're using.\n")

        outputs = tokenizer.decode(sequence)
        for b in range(bsz):
            print(f"{b + 1} {repr(prompts[b])} -> {repr(outputs[b])}")

        # TODO Save the logits and then rerun each prompt with a batch size of 1, same input. The logits should be identical.
