from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
import argparse, sys, os, glob
from torch import version as torch_version

def add_args(parser):

    parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path")
    parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)")
    parser.add_argument("-m", "--model", type = str, help = "Model weights path (.pt or .safetensors file)")
    parser.add_argument("-d", "--directory", type = str, help = "Path to directory containing config.json, model.tokenizer and * .safetensors")

    parser.add_argument("-gs", "--gpu_split", type = str, help = "Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7")
    parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length", default = 2048)
    parser.add_argument("-gpfix", "--gpu_peer_fix", action = "store_true", help = "Prevent direct copies of data between GPUs")

    parser.add_argument("-mmrt", "--matmul_recons_thd", type = int, help = "No. rows at which to use reconstruction and cuBLAS for quant matmul. 0 = never, 1 = always", default = 8)
    parser.add_argument("-fmt", "--fused_mlp_thd", type = int, help = "Maximum no. of rows for which to use fused MLP. 0 = never", default = 2)
    parser.add_argument("-sdpt", "--sdp_thd", type = int, help = "No. rows at which to switch to scaled_dot_product_attention. 0 = never, 1 = always", default = 8)
    parser.add_argument("-mmfr", "--matmul_fused_remap", action = "store_true", help = "Fuse column remapping in Q4 matmul kernel")

    parser.add_argument("-rnnh2", "--rmsnorm_no_half2", action = "store_true", help = "Don't use half2 in RMS norm kernel")
    parser.add_argument("-rpnh2", "--rope_no_half2", action = "store_true", help = "Don't use half2 in RoPE kernel")
    parser.add_argument("-mmnh2", "--matmul_no_half2", action = "store_true", help = "Don't use half2 in Q4 matmul kernel")
    parser.add_argument("-snh2", "--silu_no_half2", action = "store_true", help = "Don't use half2 in SiLU kernel")
    parser.add_argument("-nh2", "--no_half2", action = "store_true", help = "(All of the above) disable half2 in all kernela")
    parser.add_argument("-fh2", "--force_half2", action = "store_true", help = "Force enable half2 even if unsupported")


def post_parse(args):

    if args.no_half2 or torch_version.hip and not args.force_half2:
        args.rmsnorm_no_half2 = True
        args.rope_no_half2 = True
        args.matmul_no_half2 = True
        args.silu_no_half2 = True


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
    if args.gpu_split is not None: print_opts.append(f"gpu_split: {args.gpu_split}")
    if args.gpu_peer_fix: print_opts.append("gpu_peer_fix")

    if extra_options is not None: print_opts += extra_options

    print(f" -- Tokenizer: {args.tokenizer}")
    print(f" -- Model config: {args.config}")
    print(f" -- Model: {args.model}")
    print(f" -- Sequence length: {args.length}")

    print(f" -- Tuning:")
    print(f" -- --matmul_recons_thd: {args.matmul_recons_thd}" + (" (disabled)" if args.matmul_recons_thd == 0 else ""))
    print(f" -- --fused_mlp_thd: {args.fused_mlp_thd}" + (" (disabled)" if args.fused_mlp_thd == 0 else ""))
    print(f" -- --sdp_thd: {args.sdp_thd}" + (" (disabled)" if args.sdp_thd == 0 else ""))
    if args.matmul_fused_remap: print(f" -- --matmul_fused_remap")
    if args.rmsnorm_no_half2: print(f" -- --rmsnorm_no_half2")
    if args.rope_no_half2: print(f" -- --rope_no_half2")
    if args.matmul_no_half2: print(f" -- --matmul_no_half2")
    if args.silu_no_half2: print(f" -- --silu_no_half2")

    print(f" -- Options: {print_opts}")


# Build ExLlamaConfig from args

def make_config(args):

    config = ExLlamaConfig(args.config)
    config.model_path = args.model

    config.max_seq_len = args.length
    config.set_auto_map(args.gpu_split)
    config.gpu_peer_fix = args.gpu_peer_fix

    config.matmul_recons_thd = args.matmul_recons_thd
    config.fused_mlp_thd = args.fused_mlp_thd
    config.sdp_thd = args.sdp_thd
    config.matmul_fused_remap = args.matmul_fused_remap

    config.rmsnorm_no_half2 = args.rmsnorm_no_half2
    config.rope_no_half2 = args.rope_no_half2
    config.matmul_no_half2 = args.matmul_no_half2
    config.silu_no_half2 = args.silu_no_half2

    return config


# Print stats after loading model

def print_stats(model):

    print(f" -- Groupsize (inferred): {model.config.groupsize if model.config.groupsize is not None else 'None'}")
    print(f" -- Act-order (inferred): {'yes' if model.config.act_order else 'no'}")
