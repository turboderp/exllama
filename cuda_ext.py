# from abc import ABC
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load
import os
import sys

# TODO: This is a kludge to make the C++ extension load when the library is imported elsewhere. May not be needed
# with the package installed, if so maybe find better solution.

library_dir = os.path.dirname(os.path.abspath(__file__))
extension_name = "exllama_ext"

# another kludge to get things compiling in Windows
windows = os.name == "nt"
if windows:
    def find_msvc():
        for msvc_dir in [a + "\\Microsoft Visual Studio\\" + b + "\\" + c + "\\VC\Tools\\MSVC\\"
            for b in ["2022", "2019", "2017"]
            for a in [os.environ["ProgramW6432"], os.environ["ProgramFiles(x86)"]]
            for c in ["BuildTools", "Community", "Professional", "Enterprise", "Preview"]
        ]:
            if not os.path.exists(msvc_dir):
                continue
            versions = sorted(os.listdir(msvc_dir), reverse=True)
            for version in versions:
                compiler_dir = msvc_dir + version + "\\bin\\Hostx64\\x64"
                if os.path.exists(compiler_dir) and os.path.exists(compiler_dir + "\\cl.exe"):
                    return compiler_dir
        return None
    
    import subprocess
    try:
        subprocess.check_output(["where", "cl"])
    except subprocess.CalledProcessError as e:
        cl_path = find_msvc()
        if cl_path:
            print("Injected compiler path:", cl_path)
            os.environ["path"] += ";" + cl_path
        else:
            print("Unable to find cl.exe; compilation will probably fail.")

exllama_ext = load(
    name = extension_name,
    sources = [
        os.path.join(library_dir, "exllama_ext/exllama_ext.cpp"),

        os.path.join(library_dir, "exllama_ext/cuda_buffers.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4_matrix.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4_matmul.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/column_remap.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/rms_norm.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/rope.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/half_matmul.cu"),

        os.path.join(library_dir, "exllama_ext/cuda_func/q4_mlp.cu"),

        os.path.join(library_dir, "exllama_ext/cpu_func/rep_penalty.cpp")

    ],
    # verbose = True,
    # extra_cflags = ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
    extra_ldflags=["cublas.lib"] if windows else [],
)

# from exllama_ext import set_tuning_params
# from exllama_ext import prepare_buffers
from exllama_ext import make_q4
from exllama_ext import q4_matmul
from exllama_ext import half_matmul
from exllama_ext import half_matmul_cublas
from exllama_ext import q4_mlp
from exllama_ext import rms_norm
from exllama_ext import rope_
from exllama_ext import rep_penalty


# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension

none_tensor = torch.empty((1, 1), device = "meta")


# Construct Q4Matrix, return handle

def ext_make_q4(qweight, qzeros, scales, g_idx, device):

    return make_q4(qweight,
                   qzeros,
                   scales,
                   g_idx if g_idx is not None else none_tensor,
                   device)


# Matrix multiplication, returns x @ q4

def ext_q4_matmul(x, q4, q4_width):

    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype = torch.float16, device = x.device)

    q4_matmul(x, q4, output)
    return output.view(outshape)


# Matrix multiplication, returns x @ w, both half-precision tensors

def ext_half_matmul(x, w, cublas = False):

    outshape = x.shape[:-1] + (w.shape[1],)
    x = x.view(-1, x.shape[-1])

    if cublas:
        output = torch.empty((x.shape[0], w.shape[1]), dtype = torch.float16, device = x.device)
        half_matmul_cublas(x, w, output)
    else:
        output = torch.zeros((x.shape[0], w.shape[1]), dtype = torch.float16, device = x.device)
        half_matmul(x, w, output)

    return output.view(outshape)  ##


# RoPE embeddings, in_place

def ext_rope_(x, sin, cos, past_len, num_heads, head_dim):

    rope_(x, sin, cos, past_len, num_heads, head_dim)



# Llama MLP, compute: (SiLU(x @ gate_proj) * (x @ up_proj)) @ down_proj

def ext_q4_mlp(x,
               rms_norm_weight,
               epsilon,
               gate_proj,
               up_proj,
               down_proj):

    x = x.view(-1, x.shape[-1])

    q4_mlp(x,
           rms_norm_weight,
           epsilon,
           gate_proj,
           up_proj,
           down_proj)


# RMS norm: x = x * w / sqrt(row_mean(x * x) + epsilon)

def ext_rms_norm(x, w, epsilon):

    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    output = torch.empty_like(x)
    rms_norm(x, w, output, epsilon)

    return output.view(outshape)

def ext_rms_norm_(x, w, epsilon):

    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    rms_norm(x, w, x, epsilon)


# Repetition penalty

def ext_rep_penalty_mask_cpu(vocab_size, sequence, penalty_max, sustain, decay):

    rep_mask = torch.empty(vocab_size, dtype = torch.float32)
    rep_penalty(sequence, rep_mask, penalty_max, sustain, decay)
    return rep_mask
