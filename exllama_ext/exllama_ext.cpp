#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include "cpu_func/rep_penalty.h"

#include "cuda_buffers.cuh"
#include "cuda_func/column_remap.cuh"
#include "cuda_func/half_matmul.cuh"
#include "cuda_func/q4v2_matmul.cuh"
#include "cuda_func/q4v2_mlp.cuh"
#include "cuda_func/q4v2_recons.cuh"
#include "cuda_func/q4v2_sequential.cuh"
#include "cuda_func/rms_norm.cuh"
#include "cuda_func/rope.cuh"
#include "util.cuh"

// Check CUDA return code. We don't want to include Torch headers in the .cu files because parsing them adds almost a
// minute to the compile time on a 12900K. Also passing exceptions back to Python is super tricky, so in place of
// exceptions, CUDA functions return with a cudaError_t which we can parse and dump to the console.

void check_cuda(cudaError_t ret)
{
    switch (ret)
    {
        case cudaSuccess:
            break;

        case cudaUnspecified:
            printf(" **** Unspecified error\n");
            TORCH_CHECK(false, "CUDA error");
            break;

        default:
            printf(" **** CUDA error\n"); \
            printf(" **** %s\n", cudaGetErrorString(ret)); \
            TORCH_CHECK(false, "CUDA error"); \
            break;
    }
}

// Some decluttering macros

#define STRINGIFY_(__x) #__x
#define STRINGIFY(__x) STRINGIFY_(__x)
#define TORCH_CHECK_DTYPE(__x, __dtype) TORCH_CHECK((__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_DTYPE_OPT(__x, __dtype) TORCH_CHECK((__x).device().is_meta() || (__x).dtype() == torch::__dtype, #__x " is incorrect datatype, must be " #__dtype)
#define TORCH_CHECK_SHAPES(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPES_OPT(__x, __dim_x, __y, __dim_y, __scale_y) TORCH_CHECK((__x).device().is_meta() || (__x).size(__dim_x) == (__y).size(__dim_y) * __scale_y, #__x " and " #__y " have incompatible shapes")
#define TORCH_CHECK_SHAPE_MOD(__x, __dim_x, __mod) TORCH_CHECK((__x).size(__dim_x) % __mod == 0, #__x ".shape[" STRINGIFY(__dim_x) "] must be a multiple of " STRINGIFY(__mod))

#define TORCH_CHECK_DEVICE_INDEX(__index) \
do { \
    TORCH_CHECK(__index >= 0, "no device index"); \
    TORCH_CHECK(__index < CUDA_MAX_DEVICES, "invalid device index"); \
} while(0)

#define TORCH_CHECK_QUANT(__w, __w_scales, __w_zeros, __seq_g_idx, __x_map) \
do { \
    TORCH_CHECK_DTYPE(__w, kInt); \
    TORCH_CHECK_DTYPE(__w_scales, kHalf); \
    TORCH_CHECK_DTYPE(__w_zeros, kInt); \
    TORCH_CHECK_DTYPE_OPT(__seq_g_idx, kShort); \
    TORCH_CHECK_DTYPE_OPT(__x_map, kInt); \
    TORCH_CHECK_SHAPES_OPT(__seq_g_idx, 0, __w, 0, 2 * 8); \
    TORCH_CHECK_SHAPES_OPT(__x_map, 0, __w, 0, 8); \
} while(0)

int get_groupsize(torch::Tensor w, torch::Tensor w_zeros)
{
    int groupsize = w.size(0) * 8 / w_zeros.size(0);
    TORCH_CHECK(groupsize * w_zeros.size(0) == w.size(0) * 8, "w.shape[-2] must be a multiple of zeros.shape[-2]")
    return groupsize;
}

// Prepare buffers for forward pass

void prepare_buffers
(
    torch::Device device,
    int rows,
    int mlp_rows,
    int intermediate_size,
    int hidden_size
)
{
    int device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    check_cuda(
        prepare_buffers_cuda
        (
            device_index,
            rows,
            mlp_rows,
            intermediate_size,
            hidden_size
        )
    );
}

void free_buffers
(
    torch::Device device
)
{
    int device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    check_cuda(
        free_buffers_cuda(device_index)
    );
}

// Matmul half @ quant -> half

void q4v2_matmul
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor w_scales,
    torch::Tensor w_zeros,
    torch::Tensor seq_g_idx,
    torch::Tensor x_map
)
{
    TORCH_CHECK_QUANT(w, w_scales, w_zeros, seq_g_idx, x_map);
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_SHAPE_MOD(x, 1, 256);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 8);
    TORCH_CHECK_DTYPE(out, kHalf);

    int groupsize = get_groupsize(w, w_zeros);
    int height = x.size(0);
    int dim = x.size(1);
    int width = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w_scales));

    check_cuda(
        q4v2_matmul_cuda
        (
            (half*) x.data_ptr(),
            (uint32_t*) w.data_ptr(),
            (half*) out.data_ptr(),
            (half*) w_scales.data_ptr(),
            (uint32_t*) w_zeros.data_ptr(),
            height,
            dim,
            width,
            groupsize,
            seq_g_idx.device().is_meta() ? NULL : (uint16_t*) seq_g_idx.data_ptr(),
            x_map.device().is_meta() ? NULL : (uint32_t*) x_map.data_ptr()
        )
    );
}

// Reconstruct half matrix from quant

void q4v2_recons
(
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor w_scales,
    torch::Tensor w_zeros,
    torch::Tensor seq_g_idx,
    torch::Tensor x_map
)
{
    TORCH_CHECK_QUANT(w, w_scales, w_zeros, seq_g_idx, x_map);
    TORCH_CHECK_DTYPE(out, kHalf);

    int groupsize = get_groupsize(w, w_zeros);
    int height = w.size(0);
    int width = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w_scales));

    check_cuda(
        q4v2_recons_cuda
        (
            (uint32_t*) w.data_ptr(),
            (half*) out.data_ptr(),
            (half*) w_scales.data_ptr(),
            (uint32_t*) w_zeros.data_ptr(),
            height,
            width,
            groupsize,
            seq_g_idx.device().is_meta() ? NULL : (uint16_t*) seq_g_idx.data_ptr()
        )
    );
}

// Rearrange rows in w so group index is sequential, build new index and corresponding column map for matmul

void q4v2_sequential
(
    torch::Tensor w,
    torch::Tensor g_idx,        // size: w_height * 8
    torch::Tensor seq_g_idx,    // size: w_height * 8 * 2
    torch::Tensor x_map,        // size: w_height * 8
    const int num_groups
)
{
    TORCH_CHECK_DTYPE(w, kInt);
    TORCH_CHECK_DTYPE(g_idx, kInt);
    TORCH_CHECK_DTYPE(seq_g_idx, kShort);
    TORCH_CHECK_DTYPE(x_map, kInt);
    TORCH_CHECK_SHAPES(g_idx, 0, x_map, 0, 1);
    TORCH_CHECK_SHAPES(seq_g_idx, 0, g_idx, 0, 2);
    TORCH_CHECK_SHAPES(g_idx, 0, w, 0, 8);

    int height = w.size(0);
    int width = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));

    check_cuda(
        q4v2_sequential_cuda
        (
            (uint32_t*) w.data_ptr(),
            height,
            width,
            (uint32_t*) g_idx.data_ptr(),
            (uint16_t*) seq_g_idx.data_ptr(),
            (uint32_t*) x_map.data_ptr(),
            num_groups
        )
    );
}

// Matmul half @ half -> half, custom kernel

void half_matmul
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);

    int height = x.size(0);
    int dim = x.size(1);
    int width = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    check_cuda(
        half_matmul_cuda
        (
            (half*) x.data_ptr(),
            (half*) w.data_ptr(),
            (half*) out.data_ptr(),
            height,
            dim,
            width
        )
    );
}

// Matmul half @ half -> half using cuBLAS

void half_matmul_cublas
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);

    int height = x.size(0);
    int dim = x.size(1);
    int width = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();

    check_cuda(
        half_matmul_cublas_cuda
        (
            (half*) x.data_ptr(),
            (half*) w.data_ptr(),
            (half*) out.data_ptr(),
            height,
            dim,
            width,
            handle
        )
    );
}

// Remap columns in half tensor

void column_remap
(
    torch::Tensor x,
    torch::Tensor x_new,
    torch::Tensor x_map
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(x_new, kHalf);
    TORCH_CHECK_DTYPE(x_map, kInt);
    TORCH_CHECK_SHAPES(x_map, 0, x, 1, 1);

    int height = x.size(0);
    int width = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    check_cuda(
        column_remap_cuda
        (
            (half*) x.data_ptr(),
            (half*) x_new.data_ptr(),
            height,
            width,
            (uint32_t*) x_map.data_ptr()
        )
    );
}

// Llama MLP. Unfinished. Works on all models but is still 5% slower than regular MLP with quantized layers

void q4v2_mlp
(
    torch::Tensor x,                // shape == (height, dim)
    torch::Tensor out,              // shape == x.shape

    torch::Tensor rms_norm_weight,  // shape == (x.shape[1],) == (dim,)
    float epsilon,

    torch::Tensor gate,
    torch::Tensor gate_scales,
    torch::Tensor gate_zeros,
    torch::Tensor gate_seq_g_idx,
    torch::Tensor gate_x_map,

    torch::Tensor up,
    torch::Tensor up_scales,
    torch::Tensor up_zeros,
    torch::Tensor up_seq_g_idx,
    torch::Tensor up_x_map,

    torch::Tensor down,
    torch::Tensor down_scales,
    torch::Tensor down_zeros,
    torch::Tensor down_seq_g_idx,
    torch::Tensor down_x_map
)
{
    TORCH_CHECK_QUANT(gate, gate_scales, gate_zeros, gate_seq_g_idx, gate_x_map);
    TORCH_CHECK_QUANT(up, up_scales, up_zeros, up_seq_g_idx, up_x_map);
    TORCH_CHECK_QUANT(down, down_scales, down_zeros, down_seq_g_idx, down_x_map);
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(rms_norm_weight, kHalf);
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, out, 1, 1);
    TORCH_CHECK_SHAPES(x, 1, gate, 0, 8);
    TORCH_CHECK_SHAPES(x, 1, up, 0, 8);
    TORCH_CHECK_SHAPES(x, 1, down, 1, 1);
    TORCH_CHECK_SHAPES(gate, 1, down, 0, 8);
    TORCH_CHECK_SHAPE_MOD(x, 1, 256);

    int gate_groupsize = get_groupsize(gate, gate_zeros);
    int up_groupsize = get_groupsize(up, up_zeros);
    int down_groupsize = get_groupsize(down, down_zeros);
    int height = x.size(0);
    int dim = x.size(1);
    int width = gate.size(1);

    torch::Device device = x.device();
    int device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    check_cuda(
        q4v2_mlp_cuda
        (
            (half*) x.data_ptr(),
            (half*) out.data_ptr(),

            (half*) rms_norm_weight.data_ptr(),
            epsilon,

            (uint32_t*) gate.data_ptr(),
            (half*) gate_scales.data_ptr(),
            (uint32_t*) gate_zeros.data_ptr(),
            gate_seq_g_idx.device().is_meta() ? NULL : (uint16_t*) gate_seq_g_idx.data_ptr(),
            gate_x_map.device().is_meta() ? NULL : (uint32_t*) gate_x_map.data_ptr(),
            gate_groupsize,

            (uint32_t*) up.data_ptr(),
            (half*) up_scales.data_ptr(),
            (uint32_t*) up_zeros.data_ptr(),
            up_seq_g_idx.device().is_meta() ? NULL : (uint16_t*) up_seq_g_idx.data_ptr(),
            up_x_map.device().is_meta() ? NULL : (uint32_t*) up_x_map.data_ptr(),
            up_groupsize,

            (uint32_t*) down.data_ptr(),
            (half*) down_scales.data_ptr(),
            (uint32_t*) down_zeros.data_ptr(),
            down_seq_g_idx.device().is_meta() ? NULL : (uint16_t*) down_seq_g_idx.data_ptr(),
            down_x_map.device().is_meta() ? NULL : (uint32_t*) down_x_map.data_ptr(),
            down_groupsize,

            height,
            dim,
            width,

            device_index
        )
    );
}

// RMS layernorm

void rms_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out,
    float epsilon
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(w, kHalf);
    TORCH_CHECK_DTYPE(out, kHalf);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, w, 0, 1);
    TORCH_CHECK_SHAPES(x, 0, out, 0, 1);
    TORCH_CHECK_SHAPES(x, 1, out, 1, 1);

    int rows = x.size(0);
    int dim = x.size(1);

    torch::Device device = x.device();
    int device_index = device.index();
    TORCH_CHECK_DEVICE_INDEX(device_index);
    const at::cuda::OptionalCUDAGuard device_guard(device);

    check_cuda(
        rms_norm_cuda
        (
            (half*) x.data_ptr(),
            (half*) w.data_ptr(),
            (half*) out.data_ptr(),
            epsilon,
            rows,
            dim,
            device_index
        )
    );
}

// RoPE rotary positional embeddings

void rope
(
    torch::Tensor x,
    torch::Tensor sin,
    torch::Tensor cos,
    int past_len,
    int num_heads,
    int head_dim
)
{
    TORCH_CHECK_DTYPE(x, kHalf);
    TORCH_CHECK_DTYPE(sin, kHalf);
    TORCH_CHECK_DTYPE(cos, kHalf);
    TORCH_CHECK(head_dim == cos.size(-1), "cos table does not match head_dim");
    TORCH_CHECK(head_dim == sin.size(-1), "sin table does not match head_dim");

    int rows = x.numel() / head_dim;

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    check_cuda(
        rope_cuda
        (
            (half*) x.data_ptr(),
            (half*) sin.data_ptr(),
            (half*) cos.data_ptr(),
            rows,
            head_dim,
            num_heads,
            past_len
        )
    );
}

// Repetition penalty (CPU)

void rep_penalty
(
    torch::Tensor sequence,
    torch::Tensor rep_mask,
    float penalty_max,
    int sustain,
    int decay
)
{
    TORCH_CHECK_DTYPE(sequence, kLong);
    TORCH_CHECK_DTYPE(rep_mask, kFloat);

    int vocab_size = rep_mask.size(0);
    int seq_len = sequence.size(-1);

    rep_penalty_cpu
    (
        vocab_size,
        (uint64_t*) sequence.data_ptr(),
        (float*) rep_mask.data_ptr(),
        penalty_max,
        sustain,
        decay,
        seq_len
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("q4v2_matmul", &q4v2_matmul, "q4v2 matrix multiplication");
  m.def("q4v2_mlp", &q4v2_mlp, "q4v2 llama mlp");
  m.def("q4v2_recons", &q4v2_recons, "q4v2 matrix reconstruction");
  m.def("q4v2_sequential", &q4v2_sequential, "q4v2 matrix serialization");
  m.def("column_remap", &column_remap, "half matrix column remapping");
  m.def("half_matmul", &half_matmul, "half matrix multiplication");
  m.def("half_matmul_cublas", &half_matmul_cublas, "half matrix multiplication");
  m.def("rms_norm", &rms_norm, "rms norm");
  m.def("rope", &rope, "rotary position embeddings");
  m.def("rep_penalty", &rep_penalty, "repetition penalty mask");
  m.def("prepare_buffers", &prepare_buffers, "prepare buffers");
  m.def("free_buffers", &free_buffers, "free buffers");
 }
