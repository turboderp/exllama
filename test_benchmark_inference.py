from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
import time
import torch
import torch.nn.functional as F
import argparse
import json
import math
import sys

testdata_path = "testdata.jsonl"

torch.set_grad_enabled(False)
torch.cuda._lazy_init()
# torch.backends.cuda.matmul.allow_tf32 = True
torch.set_printoptions(precision = 10)
torch_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

class ModelWrapper:

    def __init__(self, tokenizer_model_path, model_config_path, model_path, attention, matmul, length, stream):

        self.tokenizer_model_path = tokenizer_model_path
        self.model_config_path = model_config_path
        self.model_path = model_path
        self.cache = None
        self.pkv = None

        config = ExLlamaConfig(model_config_path)
        config.model_path = model_path
        config.max_seq_len = length
        config.is_v1_model = False

        # config.device_map.layers[:] = ["cuda:1"] * 40
        # config.device_map.lm_head = "cuda:1"
        # config.device_map.norm = "cuda:1"

        config.stream_layer_interval = stream

        config.attention_method = attention
        config.matmul_method = matmul

        self.model = ExLlama(config)
        self.tokenizer = ExLlamaTokenizer(tokenizer_model_path)


    def begin(self):

        if self.cache is None: self.cache = ExLlamaCache(self.model)
        else: self.cache.current_seq_len = 0


    def next_logits(self, input_ids, last_id_only = True):

        return self.model.forward(input_ids, self.cache, last_id_only)


    def tokenize(self, text):

        return self.tokenizer.encode(text)


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

parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path", required = True)
parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)", required = True)
parser.add_argument("-m", "--model", type = str, help = "Model weights path (.pt or .safetensors file)", required = True)

parser.add_argument("-a", "--attention", type = ExLlamaConfig.AttentionMethod.argparse, choices = list(ExLlamaConfig.AttentionMethod), help="Attention method", default = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP)
parser.add_argument("-mm", "--matmul", type = ExLlamaConfig.MatmulMethod.argparse, choices = list(ExLlamaConfig.MatmulMethod), help="Matmul method", default = ExLlamaConfig.MatmulMethod.SWITCHED)
parser.add_argument("-s", "--stream", type = int, help = "Stream layer interval", default = 0)

parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length", default = 2048)
parser.add_argument("-p", "--perf", action = "store_true", help = "Benchmark speed and VRAM usage")
parser.add_argument("-ppl", "--perplexity", action = "store_true", help = "Perplexity benchmark (slow)")

args = parser.parse_args()

# Some feedback

print(f" -- Loading model")
print(f" -- Tokenizer: {args.tokenizer}")
print(f" -- Model config: {args.config}")
print(f" -- Model: {args.model}")
print(f" -- Sequence length: {args.length}")

print_opts = []
print_opts.append("attention: " + str(args.attention))
print_opts.append("matmul: " + str(args.matmul))
if args.perf: print_opts.append("perf")
if args.perplexity: print_opts.append("ppl")
if args.stream > 0: print_opts.append(f"stream: {args.stream}")

print(f" -- Options: {print_opts}")

# Instantiate model

wrapper = timer("Load model", lambda: ModelWrapper(args.tokenizer, args.config, args.model, args.attention, args.matmul, args.length, args.stream))

print(f" -- Groupsize (inferred): {wrapper.model.config.groupsize if wrapper.model.config.groupsize is not None else 'None'}")
print(f" -- Act-order (inferred): {'yes' if wrapper.model.config.act_order else 'no'}")

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
        sys.stdout.flush()

        logprob_sum = 0.0
        logprob_count = 0
        ex_count = 100
        for ex in ds:

            wrapper.begin()

            ids = wrapper.tokenize(ex)
            ids = ids[:, :max_seq_len + 1]
            input_ids = ids[:, :-1]
            target_ids = ids[:, 1:]

            logits = wrapper.next_logits(input_ids, last_id_only = False)

            log_probs = F.log_softmax(logits, dim = -1)
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
        print(f" ** Perplexity: {perplexity:.4f}")

