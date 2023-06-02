from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
import argparse, sys, os, glob

# Get config from command line args

def init_model():

    parser = argparse.ArgumentParser(description = "Simple web-based chatbot for ExLlama")

    parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path")
    parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)")
    parser.add_argument("-m", "--model", type = str, help = "Model weights path (.pt or .safetensors file)")
    parser.add_argument("-d", "--directory", type = str, help = "Path to directory containing config.json, model.tokenizer and * .safetensors")

    parser.add_argument("-a", "--attention", type = ExLlamaConfig.AttentionMethod.argparse, choices = list(ExLlamaConfig.AttentionMethod), help="Attention method", default = ExLlamaConfig.AttentionMethod.SWITCHED)
    parser.add_argument("-mm", "--matmul", type = ExLlamaConfig.MatmulMethod.argparse, choices = list(ExLlamaConfig.MatmulMethod), help="Matmul method", default = ExLlamaConfig.MatmulMethod.SWITCHED)
    parser.add_argument("-mlp", "--mlp", type = ExLlamaConfig.MLPMethod.argparse, choices = list(ExLlamaConfig.MLPMethod), help="Matmul method", default = ExLlamaConfig.MLPMethod.SWITCHED)
    parser.add_argument("-gs", "--gpu_split", type = str, help = "Comma-separated list of VRAM (in GB) to use per GPU device for model layers, e.g. -gs 20,7,7")
    parser.add_argument("-dq", "--dequant", type = str, help = "Number of layers (per GPU) to de-quantize at load time")

    parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length", default = 2048)

    parser.add_argument("-gpfix", "--gpu_peer_fix", action="store_true", help="Prevent direct copies of data between GPUs")

    args = parser.parse_args()

    # Get model files from --directory

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

    print_opts = []
    print_opts.append("attention: " + str(args.attention))
    print_opts.append("matmul: " + str(args.matmul))
    print_opts.append("mlp: " + str(args.mlp))
    if args.gpu_split is not None: print_opts.append(f"gpu_split: {args.gpu_split}")
    if args.dequant is not None: print_opts.append(f"dequant: {args.dequant}")
    if args.gpu_peer_fix: print_opts.append("gpu_peer_fix")

    print(f" -- Tokenizer: {args.tokenizer}")
    print(f" -- Model config: {args.config}")
    print(f" -- Model: {args.model}")
    print(f" -- Sequence length: {args.length}")
    print(f" -- Options: {print_opts}")

    # Build config

    config = ExLlamaConfig(args.config)
    config.model_path = args.model

    config.attention_method = args.attention
    config.matmul_method = args.matmul
    config.mlp_method = args.mlp
    config.gpu_peer_fix = args.gpu_peer_fix
    config.set_auto_map(args.gpu_split)
    config.set_dequant(args.dequant)
    config.max_seq_len = args.length

    print(f" -- Loading model...")
    model = ExLlama(config)

    print(f" -- Loading tokenizer...")
    tokenizer = ExLlamaTokenizer(args.tokenizer)

    print(f" -- Groupsize (inferred): {model.config.groupsize if model.config.groupsize is not None else 'None'}")
    print(f" -- Act-order (inferred): {'yes' if model.config.act_order else 'no'}")

    return model, tokenizer
