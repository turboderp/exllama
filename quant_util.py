# from abc import ABC
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load
import os

# TODO: This is a kludge to make the C++ extension load when the library is imported elsewhere. May not be needed
# with the package installed, if so maybe find better solution.

library_dir = "../exllama/"

exllama_ext = load(
    name = "exllama_ext",
    sources = [
        os.path.join(library_dir, "exllama_ext/exllama_ext.cpp"),
        # os.path.join(library_dir, "exllama_ext/exllama_ext_v1_recons.cu"),
        # os.path.join(library_dir, "exllama_ext/exllama_ext_v1_matmul.cu"),
        os.path.join(library_dir, "exllama_ext/q4v2_recons.cu"),
        os.path.join(library_dir, "exllama_ext/q4v2_matmul.cu")
    ],
    verbose = True,
    extra_cflags = ["-ftime-report"]
)

# from exllama_ext import vecquant4recons_v1
# from exllama_ext import vecquant4matmul_v1
from exllama_ext import q4v2_recons
from exllama_ext import q4v2_matmul

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


def _matmul4bit_v2(x, qweight, scales, zeros, groupsize):

    outshape = x.shape[:-1] + (qweight.shape[1],)
    x = x.view(-1, x.shape[-1])
    y = torch.zeros((x.shape[0], qweight.shape[-1]), dtype = torch.float16, device = x.device)
    q4v2_matmul(x, qweight, y, scales, zeros, groupsize)

    return y.reshape(outshape)


# def _matmul4bit_v1_recons(x, qweight, scales, zeros, transpose = False):
#
#     if not transpose: assert qweight.shape[0] * 8 == x.shape[-1]
#     else: assert qweight.shape[1] == x.shape[-1]
#
#     buffer = torch.zeros((qweight.shape[0] * 8, qweight.shape[1]), dtype = scales.dtype, device = qweight.device)
#     vecquant4recons_v1(qweight, buffer, scales, zeros)
#
#     return torch.matmul(x, buffer.T if transpose else buffer)


def _matmul4bit_v2_recons(x, qweight, scales, zeros, groupsize, transpose = False):

    if not transpose: assert qweight.shape[0] * 8 == x.shape[-1]
    else: assert qweight.shape[1] == x.shape[-1]

    buffer = torch.zeros((qweight.shape[0] * 8, qweight.shape[1]), dtype = torch.float16, device = qweight.device)
    q4v2_recons(qweight, buffer, scales, zeros, groupsize)

    return torch.matmul(x, buffer.T if transpose else buffer)


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


# Matrix multiplication, returns x @ 4-bit matrix (qweight, scales, zeros, g_idx)

def matmul4bit(x, qweight, scales, zeros, groupsize, auto_switch_thd = 8):

    # Switch over to reconstruction and PyTorch matmul for large enough matrices

    if auto_switch_thd == -1: switch = False
    elif auto_switch_thd == 0: switch = True
    else:
        xdp = 1
        for y in x.shape[:-1]: xdp *= y
        switch = (xdp > auto_switch_thd)

    # V1 weights, TODO: Test

    # if zeros.dtype != torch.int32:
    #
    #     if switch: output = _matmul4bit_v1_recons(x.to(scales.dtype), qweight, scales, zeros.float()).half()
    #     else: output = _matmul4bit_v1(x, qweight, scales, zeros.float())
    #
    # # V2 weights
    #
    # else:

    if switch: output = _matmul4bit_v2_recons(x, qweight, scales, zeros, groupsize)
    else: output = _matmul4bit_v2(x, qweight, scales, zeros, groupsize)

    return output
