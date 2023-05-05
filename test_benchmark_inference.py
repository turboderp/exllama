from model import ExLlama, ExLlamaCache, ExLlamaConfig
from autograd_ref.autograd_4bit import load_llama_model_4bit_low_ram, Autograd4bitQuantLinear
from transformers import LlamaTokenizer
import time
import torch
import torch.nn.functional as F
import argparse
import json
import math

testdata_path = "testdata.jsonl"

torch.set_grad_enabled(False)
torch.cuda._lazy_init()
# torch.backends.cuda.matmul.allow_tf32 = True
torch.set_printoptions(precision = 10)

class ModelWrapper:

    def __init__(self, tokenizer_path, model_config_path, model_path, model_groupsize, new, half, attention, matmul, length):

        self.new = new
        self.tokenizer_path = tokenizer_path
        self.model_config_path = model_config_path
        self.model_path = model_path
        self.cache = None
        self.pkv = None

        if self.new:

            config = ExLlamaConfig(model_config_path, model_path)
            config.max_seq_len = length
            config.is_v1_model = (model_groupsize == -1)
            config.groupsize = model_groupsize

            config.attention_method = args.attention
            config.matmul_method = args.matmul

            self.model = ExLlama(config)
            self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
            self.tokenizer.pad_token_id = 0
            self.tokenizer.bos_token_id = 1
            self.tokenizer.eos_token_id = 2

        else:

            self.model, self.tokenizer = load_llama_model_4bit_low_ram(tokenizer_path, model_path, groupsize = model_groupsize,  is_v1_model = (model_groupsize == -1))

            if half:

                self.model.half()
                for n, m in self.model.named_modules():
                    if isinstance(m, Autograd4bitQuantLinear):
                        if m.groupsize == -1: m.zeros = m.zeros.half()
                        m.scales = m.scales.half()
                        m.bias = m.bias.half()


    def begin(self):

        if self.new:

            if self.cache is None: self.cache = ExLlamaCache(self.model)
            else: self.cache.current_seq_len = 0

        else:

            self.pkv = None


    def next_logits(self, input_ids, last_id_only = True):

        if self.new:

            return self.model.forward(input_ids, self.cache, last_id_only)

        else:

            result = self.model.forward(input_ids, use_cache = True, past_key_values = self.pkv)
            next_logits = result["logits"]
            self.pkv = result["past_key_values"]
            return next_logits


    def tokenize(self, text):

        return self.tokenizer.encode(text, return_tensors = "pt", add_special_tokens = False)


def timer(name, func):
    t = time.time()
    ret = func()
    t = time.time() - t
    print(f" ** Time, {name}: {t:.2f} seconds")
    return ret


def mem(name, total = False):
    global mem_base, mem_last
    mem_c = torch.cuda.max_memory_allocated("cuda")
    mem_this = mem_c - mem_last if not total else mem_c - mem_base
    mem_last = mem_c
    print(f" ** VRAM, {name}: {mem_this / (1024 ** 2):,.2f} MB")


# Parse arguments

parser = argparse.ArgumentParser(description = "Benchmark tests for ExLlama")

parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer directory", required = True)
parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)", required = True)
parser.add_argument("-m", "--model", type = str, help = "Model weights path (.pt or .safetensors file)", required = True)
parser.add_argument("-g", "--groupsize", type = int, help = "Groupsize for quantized weights", default = -1)

parser.add_argument("-o", "--original", action = "store_true", help = "Use original implementation")
parser.add_argument("-half", "--half", action = "store_true", help = "Reduce to 16-bit precision (original implementation only)")
parser.add_argument("-a", "--attention", type = ExLlamaConfig.AttentionMethod.argparse, choices = list(ExLlamaConfig.AttentionMethod), help="Attention method", default = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP)
parser.add_argument("-mm", "--matmul", type = ExLlamaConfig.MatmulMethod.argparse, choices = list(ExLlamaConfig.MatmulMethod), help="Matmul method", default = ExLlamaConfig.MatmulMethod.SWITCHED)

parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length", default = 2048)
parser.add_argument("-p", "--perf", action = "store_true", help = "Benchmark speed and VRAM usage")
parser.add_argument("-ppl", "--perplexity", action = "store_true", help = "Perplexity benchmark (slow)")


args = parser.parse_args()
use_new = not args.original

# Some feedback

print(f" -- Loading model")
print(f" -- Tokenizer: {args.tokenizer}")
print(f" -- Model config: {args.config}")
print(f" -- Model: {args.model}")
print(f" -- Groupsize: {args.groupsize if args.groupsize != -1 else 'none'}")
print(f" -- Sequence length: {args.length}")

print_opts = []
if args.original:
    print_opts.append("original")
    if args.half: print_opts.append("half")
else:
    print_opts.append("attention: " + str(args.attention))
    print_opts.append("matmul: " + str(args.matmul))
if args.perf: print_opts.append("perf")
if args.perplexity: print_opts.append("ppl")

print(f" -- Options: {print_opts}")

# Instantiate model

torch.cuda.reset_peak_memory_stats("cuda")
mem_base = torch.cuda.max_memory_allocated("cuda")
mem_last = mem_base

wrapper = timer("Load model", lambda: ModelWrapper(args.tokenizer, args.config, args.model, args.groupsize, use_new, args.half, args.attention, args.matmul, args.length))

torch.cuda.reset_peak_memory_stats("cuda")
mem("Model")

# Test sequence

gen_tokens = 128
max_seq_len = args.length
ids = torch.randint(0, 31999, (1, max_seq_len - gen_tokens)).cuda()

with torch.no_grad():

    # Benchmark memory and performance

    if args.perf:

        wrapper.begin()

        t = time.time()

        print(" -- Inference, first pass.")
        logits = timer("Inference", lambda: wrapper.next_logits(ids))

        t = time.time() - t
        print(f" ** Speed: {ids.shape[-1] / t:.2f} tokens/second")

        t = time.time()

        print(f" -- Generating {gen_tokens} tokens...")
        for i in range(gen_tokens):

            logits = logits[0, -1, :]
            token = torch.argmax(logits)

            next_id = token.unsqueeze(0).unsqueeze(0)
            logits = wrapper.next_logits(next_id)

        t = time.time() - t
        print(f" ** Speed: {gen_tokens / t:.2f} tokens/second")

        mem("Inference")
        mem("Total", total = True)

    # Benchmark perplexity

    if args.perplexity:

        print(" -- Loading dataset...")

        ds = []
        with open(testdata_path) as f:
            for line in f:
                ex = json.loads(line)["text"]
                if len(ex) > 50: ds.append(ex)

        print(" -- Testing", end = "")

        logprob_sum = 0.0
        logprob_count = 0
        ex_count = 100
        for ex in ds:

            wrapper.begin()

            ids = wrapper.tokenize(ex).cuda()
            ids = ids[:, :max_seq_len + 1]
            input_ids = ids[:, :-1]
            target_ids = ids[:, 1:]

            logits = wrapper.next_logits(input_ids, last_id_only = False)

            log_probs = F.log_softmax(logits, dim = -1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            logprob_sum += token_log_probs.sum().item()
            logprob_count += target_ids.numel()

            ex_count -= 1
            if ex_count % 10 == 0: print(".", end = "")
            if ex_count == 0: break

        mean_log_prob = logprob_sum / logprob_count
        perplexity = math.exp(-mean_log_prob)

        print("")
        print(f" ** Perplexity: {perplexity:.4f}")

