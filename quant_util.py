# from abc import ABC
import torch
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.utils.cpp_extension import load
exllama_ext = load(name = "exllama_ext", sources = ["exllama_ext/exllama_ext.cpp", "exllama_ext/exllama_ext.cu"], verbose=True)

from exllama_ext import vecquant4matmul_v1
from exllama_ext import vecquant4matmul_v2
from exllama_ext import vecquant4recons_v1
from exllama_ext import vecquant4recons_v2

def _matmul4bit_v1(x, qweight, scales, zeros):

    assert qweight.shape[0] * 8 == x.shape[-1]

    outshape = x.shape[:-1] + (qweight.shape[1],)
    x = x.reshape(-1, x.shape[-1])
    y = torch.zeros((x.shape[0], qweight.shape[-1]), dtype = torch.float32, device = x.device)
    dtype = x.dtype
    x = x.half()
    vecquant4matmul_v1(x, qweight, y, scales, zeros)
    y = y.to(dtype)

    return y.reshape(outshape)


def _matmul4bit_v2(x, qweight, scales, zeros, g_idx):

    assert qweight.shape[0] * 8 == x.shape[-1]

    outshape = x.shape[:-1] + (qweight.shape[1],)
    x = x.reshape(-1, x.shape[-1])
    y = torch.zeros((x.shape[0], qweight.shape[-1]), dtype = torch.float32, device = x.device)
    dtype = x.dtype
    x = x.half()
    vecquant4matmul_v2(x, qweight, y, scales.float(), zeros, g_idx, x.shape[-1] // 2)
    y = y.to(dtype)

    return y.reshape(outshape)


def _matmul4bit_v1_recons(x, qweight, scales, zeros, transpose = False):

    if not transpose: assert qweight.shape[0] * 8 == x.shape[-1]
    else: assert qweight.shape[1] == x.shape[-1]

    buffer = torch.zeros((qweight.shape[0] * 8, qweight.shape[1]), dtype = scales.dtype, device = qweight.device)
    vecquant4recons_v1(qweight, buffer, scales, zeros)

    return torch.matmul(x, buffer.T if transpose else buffer)


def _matmul4bit_v2_recons(x, qweight, scales, zeros, g_idx, transpose=False):

    if not transpose: assert qweight.shape[0] * 8 == x.shape[-1]
    else: assert qweight.shape[1] == x.shape[-1]

    buffer = torch.zeros((qweight.shape[0] * 8, qweight.shape[1]), dtype = scales.dtype, device = qweight.device)
    vecquant4recons_v2(qweight, buffer, scales, zeros, g_idx)

    return torch.matmul(x, buffer.T if transpose else buffer)


# Backpropagation still untested.

class ExAutogradMatmul4bitCuda(torch.autograd.Function):

    # TODO: Test backpropagattion

    @staticmethod
    @custom_fwd(cast_inputs = torch.float16)  # cast_inputs is not recommended in the docs?
    def forward(ctx, x, qweight, scales, zeros, g_idx, bits, maxq):
        ctx.save_for_backward(qweight, scales, zeros, g_idx)
        if g_idx is None: output = _matmul4bit_v1_recons(x, qweight, scales, zeros)
        else: output = _matmul4bit_v2_recons(x, qweight, scales, zeros, g_idx)
        output = output.clone()
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        qweight, scales, zeros, g_idx = ctx.saved_tensors
        grad = None
        if ctx.needs_input_grad[0]:
            if g_idx is None: grad = _matmul4bit_v1_recons(grad_output, qweight, scales, zeros, transpose = True)
            else: grad = _matmul4bit_v2_recons(grad_output, qweight, scales, zeros, g_idx, transpose = True)
        return grad, None, None, None, None, None, None


# Matrix multiplication, returns x @ 4-bit matrix (qweight, scales, zeros, g_idx)

def matmul4bit(x, qweight, scales, zeros, g_idx = None, auto_switch_thd = 8):

    # Switch over to reconstruction and PyTorch matmul for large enough matrices

    if auto_switch_thd == -1: switch = False
    elif auto_switch_thd == 0: switch = True
    else:
        xdp = 1
        for y in x.shape[:-1]: xdp *= y
        switch = (xdp > auto_switch_thd)

    # V1 weights, TODO: Test

    if zeros.dtype != torch.int32:

        if switch: output = _matmul4bit_v1_recons(x.to(scales.dtype), qweight, scales, zeros.float()).half()
        else: output = _matmul4bit_v1(x, qweight, scales, zeros.float())

    # V2 weights

    else:

        if g_idx is None: g_idx = torch.zeros(qweight.shape[0] * 8, dtype = torch.int32, device = x.device)  # Hmm

        if switch: output = _matmul4bit_v2_recons(x.to(scales.dtype), qweight, scales, zeros, g_idx).half()
        else: output = _matmul4bit_v2(x, qweight, scales, zeros, g_idx)

    return output
