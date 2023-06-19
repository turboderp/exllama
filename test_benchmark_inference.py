from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
from lora import ExLlamaLora
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


def next_logits(input_ids, apply_lora, last_id_only = True):
    global model, cache

    n_logits = None
    a = 0
    while a < input_ids.shape[-1]:
        b = min(input_ids.shape[-1], a + 2048)
        n_logits = model.forward(input_ids[:, a:b], cache, last_id_only, lora = apply_lora)
        a = b

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

parser.add_argument("-p", "--perf", action = "store_true", help = "Benchmark speed and VRAM usage")
parser.add_argument("-ppl", "--perplexity", nargs = '?', const = 'default', metavar = "METHOD", help = "Perplexity benchmark (slow). Optionally specify method: default, gptq-for-llama, llama.cpp (not yet implemented)")
parser.add_argument("-ppl-ds", "--perplexity-dataset", metavar = "DATAPATH", type = str, help = "Load dataset for perplexity (JSONL if .jsonl, otherwise parses it as raw text)")
parser.add_argument("-ppl-num", "--perplexity-num", nargs = "?", type = int, help = "Number of chunks for perplexity benchmark")
parser.add_argument("-ppl-t", "--perplexity-token", action = "store_true", help = "Run perplexity test on individual tokens, for debug purposes (slow)")
parser.add_argument("-v", "--validate", action = "store_true", help = "Quick perplexity benchmark just to test if model is working at all, and short text completion")
parser.add_argument("-lora", "--lora", type = str, help = "Path to LoRA binary to use during benchmark")
parser.add_argument("-loracfg", "--lora_config", type = str, help = "Path to LoRA config to use during benchmark")

args = parser.parse_args()
model_init.post_parse(args)
model_init.get_model_files(args)

# Feedback

print_opts = []
if args.perf: print_opts.append("perf")
if args.perplexity: print_opts.append("perplexity")
if args.perplexity_dataset: print_opts.append("perplexity_dataset")
if args.perplexity_num: print_opts.append("perplexity-num")
if args.perplexity_token: print_opts.append("perplexity-token")
if args.validate: print_opts.append("validate")

model_init.print_options(args, print_opts)

# Instantiate model

config = model_init.make_config(args)

model = timer("Load model", lambda: ExLlama(config))
tokenizer = timer("Load tokenizer", lambda: ExLlamaTokenizer(args.tokenizer))

model_init.print_stats(model)

torch.cuda.reset_peak_memory_stats("cuda")
mem("Model")

# Load LoRA

lora = None
if args.lora:
    print(f" -- Loading LoRA: {args.lora}")
    if args.lora_config is None:
        print(f" ## Error: please specify lora path to adapter_config.json")
        sys.exit()
    lora = ExLlamaLora(model, args.lora_config, args.lora)

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

    # Default (legacy) method

    testdata_path = "datasets/wikitext2_val_sample.jsonl"
    testdata_context = 2048
    testdata_overlap = 0
    testdata_minlength = 50
    num_samples = 100

    # Optionally specified dataset, either raw or .jsonl

    if args.perplexity_dataset:
        testdata_path = args.perplexity_dataset
        testdata_context = 2048
        testdata_overlap = 0
        testdata_minlength = 0

    # Settings mimicking GPTQ-for-LLaMa

    if args.perplexity == "gptq-for-llama":
        if not args.perplexity_dataset:
            testdata_path = "datasets/wikitext2.txt"
        testdata_context = 2048
        testdata_overlap = 0
        testdata_minlength = 0
        num_samples = 128

    if args.perplexity == "default":
        pass

    # Overrides

    if args.perplexity_num:
        num_samples = args.perplexity_num

    print(" -- Loading dataset...")

    ppl.load(testdata_path,
             testdata_context,
             testdata_overlap,
             testdata_minlength)

    ppl.test(num_samples,
             lora = lora,
             ppl_token = args.perplexity_token)

# Validate file

if args.validate:

    ppl = Perplexity(args.perplexity, model, cache, tokenizer)

    testdata_path = "datasets/wikitext2_val_sample.jsonl"
    testdata_context = 2048
    testdata_overlap = 0
    testdata_minlength = 50

    ppl.load(testdata_path,
             testdata_context,
             testdata_overlap,
             testdata_minlength)

    begin()

    # Short perplexity tests in switched and quant mode, should produce roughly equal results

    model.config.matmul_recons_thd = 1
    ppl.test(8, lora = lora, tag = " (reconstruct)")
    model.config.matmul_recons_thd = 0
    ppl.test(8, lora = lora, tag = " (quant, token)", ppl_token = True)

    # Do a short, easy topk=1 completion to see if we're generating garbage. Should run in switched mode
    # for the prompt and quant for individual tokens

    model.config.matmul_recons_thd = 4
    generator = ExLlamaGenerator(model, tokenizer, cache)
    generator.settings.top_k = 1
    generator.lora = lora
    text = generator.generate_simple("To be or not to be, that is the", max_new_tokens = 20)
    # text = generator.generate_simple("To be or", max_new_tokens = 20)
    text = text.replace("\n", "\\n")
    print(f" ** Generation: {text}")
