from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
import argparse, sys, os, glob

def add_args(parser):

    parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path")
    parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)")
    parser.add_argument("-m", "--model", type = str, help = "Model weights path (.pt or .safetensors file)")
    parser.add_argument("-d", "--directory", type = str, help = "Path to directory containing config.json, model.tokenizer and * .safetensors")

    parser.add_argument("-a", "--attention", type = ExLlamaConfig.AttentionMethod.argparse, choices = list(ExLlamaConfig.AttentionMethod), help="Attention method", default = ExLlamaConfig.AttentionMethod.SWITCHED)
    parser.add_argument("-mlp", "--mlp", type = ExLlamaConfig.MLPMethod.argparse, choices = list(ExLlamaConfig.MLPMethod), help="Matmul method", default = ExLlamaConfig.MLPMethod.SWITCHED)

    parser.add_argument("-gs", "--gpu_split", type = str, help = "Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7")
    parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length", default = 2048)
    parser.add_argument("-gpfix", "--gpu_peer_fix", action = "store_true", help = "Prevent direct copies of data between GPUs")

    parser.add_argument("-mmrt", "--matmul_recons_thd", type = int, help = "No. rows at which to use reconstruction and cuBLAS for quant matmul. 0 = never, 1 = always", default = 8)


# Get model files from --directory

def get_model_files(args):

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


# Feedback

def print_options(args, extra_options = None):

    print_opts = []
    print_opts.append("attention: " + str(args.attention))
    print_opts.append("mlp: " + str(args.mlp))
    if args.gpu_split is not None: print_opts.append(f"gpu_split: {args.gpu_split}")
    if args.gpu_peer_fix: print_opts.append("gpu_peer_fix")

    if extra_options is not None: print_opts += extra_options

    print(f" -- Tokenizer: {args.tokenizer}")
    print(f" -- Model config: {args.config}")
    print(f" -- Model: {args.model}")
    print(f" -- Sequence length: {args.length}")
    print(f" -- Tuning:")
    print(f" -- - matmul_recons_thd: {args.matmul_recons_thd}")
    print(f" -- Options: {print_opts}")


# Build ExLlamaConfig from args

def make_config(args):

    config = ExLlamaConfig(args.config)
    config.model_path = args.model

    config.max_seq_len = args.length
    config.set_auto_map(args.gpu_split)
    config.gpu_peer_fix = args.gpu_peer_fix

    config.attention_method = args.attention
    config.mlp_method = args.mlp

    config.matmul_recons_thd = args.matmul_recons_thd

    return config


# Print stats after loading model

def print_stats(model):

    print(f" -- Groupsize (inferred): {model.config.groupsize if model.config.groupsize is not None else 'None'}")
    print(f" -- Act-order (inferred): {'yes' if model.config.act_order else 'no'}")
