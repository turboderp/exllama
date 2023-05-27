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

testdata_path = "testdata.jsonl"

torch.set_grad_enabled(False)
torch.cuda._lazy_init()
torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
torch.set_printoptions(precision = 10)
torch_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]

class ModelWrapper:

    def __init__(self, args):

        self.tokenizer_model_path = args.tokenizer
        self.model_config_path = args.config
        self.model_path = args.model
        self.cache = None
        self.pkv = None

        config = ExLlamaConfig(self.model_config_path)
        config.model_path = self.model_path
        config.max_seq_len = args.length

        # config.device_map.layers[:] = ["cuda:1"] * 40
        # config.device_map.lm_head = "cuda:1"
        # config.device_map.norm = "cuda:1"

        config.set_auto_map(args.gpu_split)
        config.set_dequant(args.dequant)
        config.stream_layer_interval = args.stream
        config.debug = args.debug
        config.gpu_peer_fix = args.gpu_peer_fix

        config.attention_method = args.attention
        config.matmul_method = args.matmul
        config.mlp_method = args.mlp

        self.model = ExLlama(config)
        self.tokenizer = ExLlamaTokenizer(self.tokenizer_model_path)


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

parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path")
parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)")
parser.add_argument("-m", "--model", type = str, help = "Model weights path (.pt or .safetensors file)")
parser.add_argument("-d", "--directory", type = str, help = "Path to directory containing config.json, model.tokenizer and * .safetensors")

parser.add_argument("-a", "--attention", type = ExLlamaConfig.AttentionMethod.argparse, choices = list(ExLlamaConfig.AttentionMethod), help="Attention method", default = ExLlamaConfig.AttentionMethod.SWITCHED)
parser.add_argument("-mm", "--matmul", type = ExLlamaConfig.MatmulMethod.argparse, choices = list(ExLlamaConfig.MatmulMethod), help="Matmul method", default = ExLlamaConfig.MatmulMethod.SWITCHED)
parser.add_argument("-mlp", "--mlp", type = ExLlamaConfig.MLPMethod.argparse, choices = list(ExLlamaConfig.MLPMethod), help="Matmul method", default = ExLlamaConfig.MLPMethod.SWITCHED)
parser.add_argument("-s", "--stream", type = int, help = "Stream layer interval", default = 0)
parser.add_argument("-gs", "--gpu_split", type = str, help = "Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7")
parser.add_argument("-dq", "--dequant", type = str, help = "Number of layers (per GPU) to de-quantize at load time")

parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length", default = 2048)
parser.add_argument("-p", "--perf", action = "store_true", help = "Benchmark speed and VRAM usage")
parser.add_argument("-ppl", "--perplexity", action = "store_true", help = "Perplexity benchmark (slow)")
parser.add_argument("-v", "--validate", action = "store_true", help = "Quick perplexity benchmark just to test if model is working at all, and short text completion")

parser.add_argument("-dbg", "--debug", action = "store_true", help = "Run debug pass")
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

print_opts = []
print_opts.append("attention: " + str(args.attention))
print_opts.append("matmul: " + str(args.matmul))
print_opts.append("mlp: " + str(args.mlp))
if args.perf: print_opts.append("perf")
if args.perplexity: print_opts.append("perplexity")
if args.validate: print_opts.append("validate")
if args.debug: print_opts.append("debug")
if args.gpu_peer_fix: print_opts.append("gpu_peer_fix")
if args.stream > 0: print_opts.append(f"stream: {args.stream}")
if args.gpu_split is not None: print_opts.append(f"gpu_split: {args.gpu_split}")
if args.dequant is not None: print_opts.append(f"dequant: {args.dequant}")

print(f" -- Options: {print_opts}")

# Instantiate model

wrapper = timer("Load model", lambda: ModelWrapper(args))

print(f" -- Groupsize (inferred): {wrapper.model.config.groupsize if wrapper.model.config.groupsize is not None else 'None'}")
print(f" -- Act-order (inferred): {'yes' if wrapper.model.config.act_order else 'no'}")

torch.cuda.reset_peak_memory_stats("cuda")
mem("Model")

# Test sequence

gen_tokens = 128
max_seq_len = args.length
ids = torch.randint(0, 31999, (1, max_seq_len - gen_tokens)).cuda()

with torch.no_grad():

    if args.debug:

        print(" !! Inference, debug pass")

        wrapper.begin()
        logits = timer("Inference", lambda: wrapper.next_logits(ids))

        wrapper.model.config.debug = False

    # Benchmark memory and performance

    if args.perf:

        wrapper.begin()

        t = time.time()

        print(" -- Inference, first pass.")
        logits = timer("Inference", lambda: wrapper.next_logits(ids))

        t = time.time() - t
        print(f" ** Speed: {ids.shape[-1] / t:.2f} tokens/second")

        for j in range(2):

            t = time.time()
            print(f" -- Generating {gen_tokens} tokens, {ids.shape[-1]} token prompt...")
            for i in range(gen_tokens):

                logits = logits[0, -1, :]
                token = torch.argmax(logits)
                next_id = token.unsqueeze(0).unsqueeze(0)
                logits = wrapper.next_logits(next_id)

            t = time.time() - t
            print(f" ** Speed: {gen_tokens / t:.2f} tokens/second")

            ids = ids[:, :4]
            wrapper.cache.current_seq_len = 4

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

                wrapper.begin()

                ids = wrapper.tokenize(ex)
                ids = ids[:, :max_seq_len + 1]
                input_ids = ids[:, :-1]
                target_ids = ids[:, 1:]

                logits = wrapper.next_logits(input_ids, last_id_only=False)

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

            wrapper.model.config.matmul_method = ExLlamaConfig.MatmulMethod.SWITCHED
            _ppl_test(" (switched)", 8)
            wrapper.model.config.matmul_method = ExLlamaConfig.MatmulMethod.QUANT_ONLY
            _ppl_test(" (quant_only)", 8)

            # Do a short, easy topk=1 completion to see if we're generating garbage. Should run in switched mode
            # for the prompt and quant for individual tokens

            wrapper.model.config.matmul_method = ExLlamaConfig.MatmulMethod.SWITCHED
            generator = ExLlamaGenerator(wrapper.model, wrapper.tokenizer, wrapper.cache)
            generator.settings.top_k = 1
            text = generator.generate_simple("To be or not to be, that is the", max_new_tokens = 20)
            text = text.replace("\n", "\\n")
            print(f" ** Generation: {text}")


