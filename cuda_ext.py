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
        os.path.join(library_dir, "exllama_ext/cuda_buffers.cu"),
        os.path.join(library_dir, "exllama_ext/cpu_func/rep_penalty.cpp"),
        os.path.join(library_dir, "exllama_ext/cuda_func/column_remap.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/half_matmul.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4v2_matmul.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4v2_mlp.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4v2_recons.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/q4v2_sequential.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/rms_norm.cu"),
        os.path.join(library_dir, "exllama_ext/cuda_func/rope.cu"),
        os.path.join(library_dir, "exllama_ext/exllama_ext.cpp")
    ],
    # verbose = True,
    # extra_cflags = ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
)

from exllama_ext import prepare_buffers

from exllama_ext import column_remap
from exllama_ext import half_matmul
from exllama_ext import half_matmul_cublas
from exllama_ext import q4v2_matmul
from exllama_ext import q4v2_mlp
from exllama_ext import q4v2_recons
from exllama_ext import q4v2_sequential
from exllama_ext import rms_norm
from exllama_ext import rope

from exllama_ext import rep_penalty


# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension

none_tensor = torch.empty((1, 1), device = "meta")


# Buffers for forward pass
# TODO: This should pass a handle to the ExLlama object so we can allocate one set of buffers per instance. Currently
# only supports one set of buffers globally

def prepare_cuda_buffers(device, temp_state, temp_mlp, temp_rms_norm):

    prepare_buffers(device, temp_state, temp_mlp, temp_rms_norm)


def _matmul_q4v2_matmul(x, w, scales, zeros, seq_g_idx, x_map):

    if x_map is not None:

        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x_mapped = torch.empty_like(x)
        column_remap(x, x_mapped, x_map)
        x = x_mapped.view(x_shape)  ##

    outshape = x.shape[:-1] + (w.shape[1],)
    x = x.view(-1, x.shape[-1])
    output = torch.empty((x.shape[0], w.shape[-1]), dtype = torch.float16, device = x.device)

    # We could pass x_map here instead of allocating a temporary tensor, but it's weirdly slow to call column_remap
    # directly, presumably due to the memory allocation. Torch is probably using a cache of buffers for the allocation
    # above.

    q4v2_matmul(x,
                w,
                output,
                scales,
                zeros,
                seq_g_idx if seq_g_idx is not None else none_tensor,
                none_tensor)

    return output.view(outshape)  ##


def _matmul_q4v2_recons(x, w, scales, zeros, seq_g_idx, x_map):

    assert w.shape[0] * 8 == x.shape[-1]

    qweight_recons = torch.empty((w.shape[0] * 8, w.shape[1]), dtype = torch.float16, device = w.device)
    q4v2_recons(w, qweight_recons, scales, zeros, seq_g_idx if seq_g_idx is not None else none_tensor, x_map if x_map is not None else none_tensor)

    # if buffer.shape[-1] > 10000: _dump_tensor(buffer, "cuda_test/model.layers.0.mlp.gate_proj.recons")

    if x_map is not None:

        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x_mapped = torch.empty_like(x)
        column_remap(x, x_mapped, x_map)
        x = x_mapped.view(x_shape)  ##

    # output = torch.matmul(x, qweight_recons)
    output = matmul_half(x, qweight_recons, cublas = True)

    return output


# Reconstruct fp16 matrix from 4-bit matrix

def dequantize_q4v2(quant_args):

    w = quant_args["qweight"]
    scales = quant_args["scales"]
    zeros = quant_args["zeros"]
    seq_g_idx = quant_args["seq_g_idx"]
    x_map = quant_args["x_map"]

    qweight_recons = torch.empty((w.shape[0] * 8, w.shape[1]), dtype = torch.float16, device = w.device)
    q4v2_recons(w, qweight_recons, scales, zeros, seq_g_idx if seq_g_idx is not None else none_tensor, none_tensor)

    if x_map is not None:

        # Rows will have already been rearranged to sequentialize the groups, so undo that

        inverse_x_map = torch.argsort(x_map)
        qweight_recons = qweight_recons[inverse_x_map]

    return qweight_recons


# Matrix multiplication, returns x @ w, both half-precision tensors

def matmul_half(x, w, cublas = False):

    outshape = x.shape[:-1] + (w.shape[1],)
    x = x.view(-1, x.shape[-1])

    if cublas:
        output = torch.empty((x.shape[0], w.shape[1]), dtype=torch.float16, device=x.device)
        half_matmul_cublas(x, w, output)
    else:
        output = torch.zeros((x.shape[0], w.shape[1]), dtype=torch.float16, device=x.device)
        half_matmul(x, w, output)

    return output.view(outshape)  ##


# Matrix multiplication, returns x @ 4-bit matrix (qweight, scales, zeros, g_idx)

def matmul_q4v2(x, quant_args, switch):

    w = quant_args["qweight"]
    scales = quant_args["scales"]
    zeros = quant_args["zeros"]
    seq_g_idx = quant_args["seq_g_idx"]
    x_map = quant_args["x_map"]

    if switch: output = _matmul_q4v2_recons(x, w, scales, zeros, seq_g_idx, x_map)
    else: output = _matmul_q4v2_matmul(x, w, scales, zeros, seq_g_idx, x_map)

    return output


# RoPE embeddings, in_place

def rope_(x, sin, cos, past_len, num_heads, head_dim):

    assert past_len + x.shape[-2] <= sin.shape[-2]
    rope(x, sin, cos, past_len, num_heads, head_dim)


# Sequentialize groups

def sequential_q4v2(w, g_idx, num_groups):

    seq_g_idx = torch.zeros((w.shape[0] * 8 * 2,), dtype = torch.short, device = w.device)
    x_map = torch.zeros_like(g_idx)

    q4v2_sequential(w, g_idx, seq_g_idx, x_map, num_groups)

    return seq_g_idx, x_map


# Llama MLP, compute: (SiLU(x @ gate_proj) * (x @ up_proj)) @ down_proj

def mlp_q4v2(x,
             rms_norm_weight,
             epsilon,
             gate_proj,
             up_proj,
             down_proj,
             intermediate_size):

    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    out = torch.empty_like(x)

    # TODO: A second buffer for the down projection shouldn't be needed since multiplying in-place without zeroing the
    # input buffer should have the same effect as adding the residual connection. Except the matmul goes crazy when the
    # output buffer isn't initialized to zeros. Could be an fp16 rounding issue. (?)

    q4v2_mlp(x,
             out,
             rms_norm_weight,
             epsilon,
             gate_proj["qweight"],
             gate_proj["scales"],
             gate_proj["zeros"],
             gate_proj["seq_g_idx"] if gate_proj["seq_g_idx"] is not None else none_tensor,
             gate_proj["x_map"] if gate_proj["x_map"] is not None else none_tensor,
             up_proj["qweight"],
             up_proj["scales"],
             up_proj["zeros"],
             up_proj["seq_g_idx"] if up_proj["seq_g_idx"] is not None else none_tensor,
             up_proj["x_map"] if up_proj["x_map"] is not None else none_tensor,
             down_proj["qweight"],
             down_proj["scales"],
             down_proj["zeros"],
             down_proj["seq_g_idx"] if down_proj["seq_g_idx"] is not None else none_tensor,
             down_proj["x_map"] if down_proj["x_map"] is not None else none_tensor)

    return out.view(outshape)


# RMS norm: x = x * w / sqrt(row_mean(x * x) + epsilon)

def llama_rms_norm(x, w, epsilon):

    outshape = x.shape
    x = x.view(-1, x.shape[-1])
    # scratch = torch.empty((x.shape[0],), dtype = torch.float32, device = x.device)
    output = torch.empty_like(x)

    # rms_norm(x, w, output, scratch, epsilon)
    rms_norm(x, w, output, epsilon)

    return output.view(outshape)


# Repetition penalty

def rep_penalty_mask_cpu(vocab_size, sequence, penalty_max, sustain, decay):

    rep_mask = torch.empty(vocab_size, dtype = torch.float32)
    rep_penalty(sequence, rep_mask, penalty_max, sustain, decay)
    return rep_mask


# Backpropagation still untested. Must be very broken at this point

class ExAutogradMatmul4bitCuda(torch.autograd.Function):

    # TODO: Test backpropagattion

    @staticmethod
    @custom_fwd(cast_inputs = torch.float16)  # cast_inputs is not recommended in the docs?
    def forward(ctx, x, qweight, scales, zeros, g_idx, bits, maxq):
        raise ValueError("Not implemented yet")
        ctx.save_for_backward(qweight, scales, zeros, g_idx)
        # if g_idx is None: output = _matmul4bit_v1_recons(x, qweight, scales, zeros)
        # else:
        output = _matmul4bit_v2_recons(x, qweight, scales, zeros, g_idx)
        output = output.clone()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        raise ValueError("Not implemented yet")
        qweight, scales, zeros, g_idx = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
        # if g_idx is None: grad = _matmul4bit_v1_recons(grad_output, qweight, scales, zeros, transpose = True)
        # else:
            grad = _matmul4bit_v2_recons(grad_output, qweight, scales, zeros, g_idx, transpose = True)
        return grad, None, None, None, None, None, None
