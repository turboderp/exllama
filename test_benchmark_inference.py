from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
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

testdata_path = "testdata.jsonl"

torch.set_grad_enabled(False)
torch.cuda._lazy_init()
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_printoptions(precision = 10)
torch_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

cache = None
model = None

def begin():
    global model, cache

    if cache is None: cache = ExLlamaCache(model)
    else: cache.current_seq_len = 0


def next_logits(input_ids, last_id_only = True):
    global model, cache

    return model.forward(input_ids, cache, last_id_only)


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
parser.add_argument("-ppl", "--perplexity", action = "store_true", help = "Perplexity benchmark (slow)")
parser.add_argument("-v", "--validate", action = "store_true", help = "Quick perplexity benchmark just to test if model is working at all, and short text completion")
parser.add_argument("-dbg", "--debug", action = "store_true", help = "Run debug pass")

args = parser.parse_args()
model_init.post_parse(args)
model_init.get_model_files(args)

# Feedback

print_opts = []
if args.perf: print_opts.append("perf")
if args.perplexity: print_opts.append("perplexity")
if args.validate: print_opts.append("validate")
if args.debug: print_opts.append("debug")

model_init.print_options(args, print_opts)

# Instantiate model

config = model_init.make_config(args)
config.debug = args.debug

model = timer("Load model", lambda: ExLlama(config))
tokenizer = timer("Load tokenizer", lambda: ExLlamaTokenizer(args.tokenizer))

model_init.print_stats(model)

torch.cuda.reset_peak_memory_stats("cuda")
mem("Model")

# Test sequence

gen_tokens = 128
max_seq_len = args.length
ids = torch.randint(0, 31999, (1, max_seq_len - gen_tokens)).cuda()

with torch.no_grad():

    if args.debug:

        print(" !! Inference, debug pass")

        begin()
        logits = timer("Inference", lambda: next_logits(ids))

        model.config.debug = False

    # Benchmark memory and performance

    if args.perf:

        # Warming up apparently makes a huge difference

        for i in range(1, 4):
            print(f" -- Warmup pass {i}...")
            begin()
            logits = timer("Warmup", lambda: next_logits(ids))

        # Do the actual benchmark

        begin()

        t = time.time()

        print(" -- Inference, first pass.")
        logits = timer("Inference", lambda: next_logits(ids))

        t = time.time() - t
        print(f" ** Speed: {ids.shape[-1] / t:.2f} tokens/second")

        for j in range(2):

            t = time.time()
            print(f" -- Generating {gen_tokens} tokens, {ids.shape[-1]} token prompt...")
            for i in range(gen_tokens):

                logits = logits[0, -1, :]
                token = torch.argmax(logits)
                next_id = token.unsqueeze(0).unsqueeze(0)
                logits = next_logits(next_id)

            t = time.time() - t
            print(f" ** Speed: {gen_tokens / t:.2f} tokens/second")

            ids = ids[:, :4]
            cache.current_seq_len = 4

        mem("Inference")
        mem("Total", total = True)

    # Benchmark perplexity

    if args.perplexity or args.validate:

        print(" -- Loading dataset...")

        ds = []
        with open(testdata_path) as f:
            for line in f:
                example = json.loads(line)["text"]
                if len(example) > 50: ds.append(example)

        def _ppl_test(text, ex_count):

            print(" -- Testing", end="")
            sys.stdout.flush()

            logprob_sum = 0.0
            logprob_count = 0

            for ex in ds:

                begin()

                ids = tokenize(ex)
                ids = ids[:, :max_seq_len + 1]
                input_ids = ids[:, :-1]
                target_ids = ids[:, 1:]

                logits = next_logits(input_ids, last_id_only=False)

                log_probs = F.log_softmax(logits, dim=-1)
                token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

                logprob_sum += token_log_probs.sum().item()
                logprob_count += target_ids.numel()

                ex_count -= 1
                if ex_count % 10 == 0:
                    print(".", end = "")
                sys.stdout.flush()
                if ex_count == 0: break

            mean_log_prob = logprob_sum / logprob_count
            perplexity = math.exp(-mean_log_prob)

            print("")
            print(f" ** Perplexity{text}: {perplexity:.4f}")

        if args.perplexity:

            _ppl_test("", 100)

        if args.validate:

            # Short perplexity tests in switched and quant mode, should produce roughly equal results

            model.config.matmul_recons_thd = 1
            _ppl_test(" (reconstruct)", 8)
            model.config.matmul_recons_thd = 0
            _ppl_test(" (quant)", 8)

            # Do a short, easy topk=1 completion to see if we're generating garbage. Should run in switched mode
            # for the prompt and quant for individual tokens

            model.config.matmul_recons_thd = 4
            generator = ExLlamaGenerator(model, tokenizer, cache)
            generator.settings.top_k = 1
            text = generator.generate_simple("To be or not to be, that is the", max_new_tokens = 20)
            text = text.replace("\n", "\\n")
            print(f" ** Generation: {text}")


