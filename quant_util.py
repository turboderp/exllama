# from abc import ABC
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load
import os

# TODO: This is a kludge to make the C++ extension load when the library is imported elsewhere. May not be needed
# with the package installed, if so maybe find better solution.

# TODO: All the CUDA stuff needs some cleanup

library_dir = "../exllama/"
extension_name = "exllama_ext"

exllama_ext = load(
    name = extension_name,
    sources = [
        os.path.join(library_dir, "exllama_ext/exllama_ext.cpp"),
        # os.path.join(library_dir, "exllama_ext/exllama_ext_v1_recons.cu"),
        # os.path.join(library_dir, "exllama_ext/exllama_ext_v1_matmul.cu"),
        os.path.join(library_dir, "exllama_ext/q4v2_recons.cu"),
        os.path.join(library_dir, "exllama_ext/q4v2_matmul.cu"),
        os.path.join(library_dir, "exllama_ext/q4v2_sequential.cu"),
        os.path.join(library_dir, "exllama_ext/column_remap.cu")
    ],
    # verbose = True,
    # extra_cflags = ["-ftime-report", "-DTORCH_USE_CUDA_DSA"]
)

# from exllama_ext import vecquant4recons_v1
# from exllama_ext import vecquant4matmul_v1
from exllama_ext import q4v2_recons
from exllama_ext import q4v2_matmul
from exllama_ext import q4v2_sequential
from exllama_ext import column_remap

# Dummy tensor to pass instead of g_idx since there is no way to pass "None" to a C++ extension

none_tensor = torch.empty((1, 1), device = "meta")

def _dump_tensor(t, name):

    t.cpu().numpy().tofile(name)


def _matmul_q4v2_matmul(x, w, scales, zeros, seq_g_idx, x_map):

    if x_map is not None:

        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x_mapped = torch.empty_like(x)
        column_remap(x, x_mapped, x_map)
        x = x_mapped.reshape(x_shape)

    outshape = x.shape[:-1] + (w.shape[1],)
    x = x.view(-1, x.shape[-1])
    output = torch.zeros((x.shape[0], w.shape[-1]), dtype = torch.float16, device = x.device)

    # We can pass x_map here instead of allocating a temporary tensor, but it's weirdly slow to call column_remap
    # directly, for some reason. TODO: Figure out the reason.

    q4v2_matmul(x,
                w,
                output,
                scales,
                zeros,
                seq_g_idx if seq_g_idx is not None else none_tensor,
                none_tensor)

    return output.reshape(outshape)


def _matmul_q4v2_recons(x, w, scales, zeros, seq_g_idx, x_map, transpose = False):

    if not transpose: assert w.shape[0] * 8 == x.shape[-1]
    else: assert w.shape[1] == x.shape[-1]

    qweight_recons = torch.empty((w.shape[0] * 8, w.shape[1]), dtype = torch.float16, device = w.device)
    q4v2_recons(w, qweight_recons, scales, zeros, seq_g_idx if seq_g_idx is not None else none_tensor)

    # if buffer.shape[-1] > 10000: _dump_tensor(buffer, "cuda_test/model.layers.0.mlp.gate_proj.recons")

    if x_map is not None:

        x_shape = x.shape
        x = x.view(-1, x.shape[-1])
        x_mapped = torch.empty_like(x)
        column_remap(x, x_mapped, x_map)
        x = x_mapped.reshape(x_shape)

    output = torch.matmul(x, qweight_recons.T if transpose else qweight_recons)

    return output


# Matrix multiplication, returns x @ 4-bit matrix (qweight, scales, zeros, g_idx)

def matmul_q4v2(x, w, scales, zeros, seq_g_idx, x_map, auto_switch_thd = 8):

    # Switch over to reconstruction and PyTorch matmul for tall enough left-hand matrices

    if auto_switch_thd == -1: switch = False
    elif auto_switch_thd == 0: switch = True
    else:
        xdp = 1
        for y in x.shape[:-1]: xdp *= y
        switch = (xdp > auto_switch_thd)

    if switch: output = _matmul_q4v2_recons(x, w, scales, zeros, seq_g_idx, x_map)
    else: output = _matmul_q4v2_matmul(x, w, scales, zeros, seq_g_idx, x_map)

    return output


def sequential_q4v2(w, g_idx, num_groups):

    seq_g_idx = torch.zeros((w.shape[0] * 8 * 2,), dtype = torch.short, device = w.device)
    x_map = torch.zeros_like(g_idx)

    q4v2_sequential(w, g_idx, seq_g_idx, x_map, num_groups)

    return seq_g_idx, x_map





    # V1 weights,

    # if zeros.dtype != torch.int32:
    #
    #     if switch: output = _matmul4bit_v1_recons(x.to(scales.dtype), qweight, scales, zeros.float()).half()
    #     else: output = _matmul4bit_v1(x, qweight, scales, zeros.float())
    #
    # # V2 weights
    #
    # else:

# TODO: Implement these

# def _matmul4bit_v1(x, qweight, scales, zeros):
#
#     assert qweight.shape[0] * 8 == x.shape[-1]
#
#     outshape = x.shape[:-1] + (qweight.shape[1],)
#     x = x.view(-1, x.shape[-1])
#     y = torch.zeros((x.shape[0], qweight.shape[-1]), dtype = torch.float32, device = x.device)
#     dtype = x.dtype
#     x = x.half()
#     vecquant4matmul_v1(x, qweight, y, scales, zeros)
#     y = y.to(dtype)
#
#     return y.reshape(outshape)

# def _matmul4bit_v1_recons(x, qweight, scales, zeros, transpose = False):
#
#     if not transpose: assert qweight.shape[0] * 8 == x.shape[-1]
#     else: assert qweight.shape[1] == x.shape[-1]
#
#     buffer = torch.zeros((qweight.shape[0] * 8, qweight.shape[1]), dtype = scales.dtype, device = qweight.device)
#     vecquant4recons_v1(qweight, buffer, scales, zeros)
#
#     return torch.matmul(x, buffer.T if transpose else buffer)


# Backpropagation still untested.

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
