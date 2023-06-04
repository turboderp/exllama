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
)

from exllama_ext import prepare_buffers
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


# Buffers for forward pass
# TODO: This should pass a handle to the ExLlama object so we can allocate one set of buffers per instance. Currently
# only supports one set of buffers globally

def ext_prepare_cuda_buffers(device, temp_state, temp_mlp, temp_rms_norm, temp_dq):

    prepare_buffers(device, temp_state, temp_mlp, temp_rms_norm, temp_dq)


# Construct Q4Matrix, return handle

def ext_make_q4(qweight, qzeros, scales, g_idx, device):

    return make_q4(qweight,
                   qzeros,
                   scales,
                   g_idx if g_idx is not None else none_tensor,
                   device)


# Matrix multiplication, returns x @ q4

def ext_q4_matmul(x, q4, q4_width, recons_thd):

    outshape = x.shape[:-1] + (q4_width,)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], q4_width), dtype = torch.float16, device = x.device)

    q4_matmul(x, q4, output, recons_thd)
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

    assert past_len + x.shape[-2] <= sin.shape[-2]
    rope_(x, sin, cos, past_len, num_heads, head_dim)



# Llama MLP, compute: (SiLU(x @ gate_proj) * (x @ up_proj)) @ down_proj

def ext_q4_mlp(x,
               rms_norm_weight,
               epsilon,
               gate_proj,
               up_proj,
               down_proj):

    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    out = torch.empty_like(x)

    # out2 = torch.empty((x.shape[0], 11008), dtype = torch.float16, device = x.device)

    # TODO: A second buffer for the down projection shouldn't be needed since multiplying in-place without zeroing the
    # input buffer should have the same effect as adding the residual connection. Except the matmul goes crazy when the
    # output buffer isn't initialized to zeros. Could be an fp16 rounding issue. (?)

    q4_mlp(x,
           out,
           rms_norm_weight,
           epsilon,
           gate_proj,
           up_proj,
           down_proj)

    return out.view(outshape)


# RMS norm: x = x * w / sqrt(row_mean(x * x) + epsilon)

def ext_rms_norm(x, w, epsilon):

    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    # scratch = torch.empty((x.shape[0],), dtype = torch.float32, device = x.device)
    output = torch.empty_like(x)

    # rms_norm(x, w, output, scratch, epsilon)
    rms_norm(x, w, output, epsilon)

    return output.view(outshape)


# Repetition penalty

def ext_rep_penalty_mask_cpu(vocab_size, sequence, penalty_max, sustain, decay):

    rep_mask = torch.empty(vocab_size, dtype = torch.float32)
    rep_penalty(sequence, rep_mask, penalty_max, sustain, decay)
    return rep_mask
