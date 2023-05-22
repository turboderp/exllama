#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

#include "column_remap.h"
#include "q4v2_matmul.h"
#include "q4v2_mlp.h"
#include "q4v2_recons.h"
#include "q4v2_sequential.h"
#include "rms_norm.h"
#include "util.h"


// Wrapper macro to handle errors between C++ and CUDA contexts. We don't want to include Torch headers in the .cu
// files because parsing them adds almost a minute to the compile time on a 12900K. Also passing exceptions back to
// Python is super tricky, so in place of proper exceptions, CUDA functions return with a cudaError_t which we can
// parse and dump to the console.

#define _cuda_raise(fn) \
do { \
    cudaError_t _cuda_err_temp; \
    {_cuda_err_temp = fn;} \
    if (_cuda_err_temp != cudaSuccess) \
    { \
        if (_cuda_err_temp == cudaUnspecified) \
        { \
            printf(" **** Unspecified error\n"); \
            printf(" **** %s in %s line $i\n", __func__, __FILE__, __LINE__); \
            TORCH_CHECK(false, "CUDA error"); \
        } \
        else \
        { \
            printf(" **** CUDA error\n"); \
            printf(" **** %s\n", cudaGetErrorString(_cuda_err_temp)); \
            printf(" **** %s in %s line $i\n", __func__, __FILE__, __LINE__); \
            TORCH_CHECK(false, "CUDA error"); \
        } \
    } \
} while(false)


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
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be a half tensor");
    TORCH_CHECK(w.dtype() == torch::kInt, "w must be an int (q4) tensor");
    TORCH_CHECK(out.dtype() == torch::kHalf, "out must be a half tensor");
    TORCH_CHECK(w_scales.dtype() == torch::kHalf, "w_scales must be a half tensor");
    TORCH_CHECK(w_zeros.dtype() == torch::kInt, "w_zeros must be an int (q4) tensor");

    TORCH_CHECK(x.size(1) == w.size(0) * 8, "x and w have incompatible shapes");
    TORCH_CHECK(x.size(1) % 256 == 0, "x.shape[1] must be multiple of 256");
    TORCH_CHECK(seq_g_idx.device().is_meta() || seq_g_idx.size(0) == w.size(0) * 2 * 8, "seq_g_idx and w have incompatible shapes");
    TORCH_CHECK(x_map.device().is_meta() || x_map.size(0) == w.size(0) * 8, "x_map and w have incompatible shapes");

    int groupsize = w.size(0) * 8 / w_zeros.size(0);
    TORCH_CHECK(groupsize * w_zeros.size(0) == w.size(0) * 8, "w.shape[-2] must be a multiple of zeros.shape[-2]")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w_scales));

    int height = x.size(0);
    int dim = x.size(1);
    int width = w.size(1);

    _cuda_raise(
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


void q4v2_recons
(
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor w_scales,
    torch::Tensor w_zeros,
    torch::Tensor seq_g_idx
)
{
    TORCH_CHECK(w.dtype() == torch::kInt, "w must be an int (q4) tensor");
    TORCH_CHECK(out.dtype() == torch::kHalf, "out must be a half tensor");
    TORCH_CHECK(w_scales.dtype() == torch::kHalf, "w_scales must be a half tensor");
    TORCH_CHECK(w_zeros.dtype() == torch::kInt, "w_zeros must be an int tensor");

    int groupsize = w.size(0) * 8 / w_zeros.size(0);
    TORCH_CHECK(groupsize * w_zeros.size(0) == w.size(0) * 8, "w.shape[0] must be a multiple of zeros.shape[0]")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w_scales));

    int height = w.size(0);
    int width = w.size(1);

    _cuda_raise(
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


void q4v2_sequential
(
    torch::Tensor w,
    torch::Tensor g_idx,        // size: w_height * 8
    torch::Tensor seq_g_idx,    // size: w_height * 8 * 2
    torch::Tensor x_map,        // size: w_height * 8
    const int num_groups
)
{
    TORCH_CHECK(w.dtype() == torch::kInt, "w must be an int (q4) tensor");
    TORCH_CHECK(g_idx.dtype() == torch::kInt, "g_idx must be an int tensor");
    TORCH_CHECK(seq_g_idx.dtype() == torch::kShort, "seq_g_idx must be a short tensor");
    TORCH_CHECK(x_map.dtype() == torch::kInt, "x_map must be an int tensor");

    TORCH_CHECK(g_idx.size(0) == x_map.size(0), "x_map must be same shape as g_idx");
    TORCH_CHECK(seq_g_idx.size(0) == g_idx.size(0) * 2, "seq_g_idx must be twice as wide as g_idx");
    TORCH_CHECK(g_idx.size(0) == w.size(0) * 8, "g_idx and w have incompatible shapes");

    int height = w.size(0);
    int width = w.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w));

    _cuda_raise(
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


void column_remap
(
    torch::Tensor x,
    torch::Tensor x_new,
    torch::Tensor x_map
)
{
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be a half tensor");
    TORCH_CHECK(x_new.dtype() == torch::kHalf, "x_new must be a half tensor");
    TORCH_CHECK(x_map.dtype() == torch::kInt, "x_map must be an int tensor");

    TORCH_CHECK(x_map.size(0) == x.size(1), "x_map and x have incompatible shapes");

    int height = x.size(0);
    int width = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    _cuda_raise(
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


// Unfinished. Works for no-act-order models but is still 5% slower than regular MLP with quantized layers

void q4v2_mlp
(
    torch::Tensor x,                // shape == (height, dim)

    torch::Tensor x_temp,           // shape == x.shape
    torch::Tensor x_col_temp,       // shape == (x.shape[0],) == (height,)
    torch::Tensor x_act_temp,       // shape == (x.shape[0], gate.shape[1]) == (height, width)

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
    TORCH_CHECK(x.dtype()                   == torch::kHalf,    "x must be a half tensor");
    TORCH_CHECK(x_temp.dtype()              == torch::kHalf,    "x_temp must be a half tensor");
    TORCH_CHECK(x_col_temp.dtype()          == torch::kFloat,   "x_col_temp must be a float tensor");
    TORCH_CHECK(x_act_temp.dtype()          == torch::kHalf,    "x_act_temp must be a half tensor");
    TORCH_CHECK(rms_norm_weight.dtype()     == torch::kHalf,    "rms_norm_weight must be a half tensor");

    TORCH_CHECK(gate.dtype()                == torch::kInt,     "gate must be an int (q4) tensor");
    TORCH_CHECK(gate_scales.dtype()         == torch::kHalf,    "gate_scales must be a half tensor");
    TORCH_CHECK(gate_zeros.dtype()          == torch::kInt,     "gate_zeros must be an int (q4) tensor");
    TORCH_CHECK(gate_seq_g_idx.device().is_meta()   || gate_seq_g_idx.dtype()      == torch::kShort,   "gate_seq_idx must be a short tensor");
    TORCH_CHECK(gate_x_map.device().is_meta()       || gate_x_map.dtype()          == torch::kInt,     "gate_x_map must be an int tensor");

    TORCH_CHECK(up.dtype()                  == torch::kInt,     "up must be an int (q4) tensor");
    TORCH_CHECK(up_scales.dtype()           == torch::kHalf,    "up_scales must be a half tensor");
    TORCH_CHECK(up_zeros.dtype()            == torch::kInt,     "up_zeros must be an int (q4) tensor");
    TORCH_CHECK(up_seq_g_idx.device().is_meta()     || up_seq_g_idx.dtype()        == torch::kShort,   "up_seq_idx must be a short tensor");
    TORCH_CHECK(up_x_map.device().is_meta()         || up_x_map.dtype()            == torch::kInt,     "up_x_map must be an int tensor");

    TORCH_CHECK(down.dtype()                == torch::kInt,     "down must be an int (q4) tensor");
    TORCH_CHECK(down_scales.dtype()         == torch::kHalf,    "down_scales must be a half tensor");
    TORCH_CHECK(down_zeros.dtype()          == torch::kInt,     "down_zeros must be an int (q4) tensor");
    TORCH_CHECK(down_seq_g_idx.device().is_meta()   || down_seq_g_idx.dtype()      == torch::kShort,   "down_seq_idx must be a short tensor");
    TORCH_CHECK(down_x_map.device().is_meta()       || down_x_map.dtype()          == torch::kInt,     "down_x_map must be an int tensor");

    TORCH_CHECK(x.size(1) == gate.size(0) * 8,      "x and gate have incompatible shapes");
    TORCH_CHECK(x.size(1) == up.size(0) * 8,        "x and up have incompatible shapes");
    TORCH_CHECK(x.size(1) == down.size(1),          "x and down have incompatible shapes");
    TORCH_CHECK(gate.size(1) == down.size(0) * 8,   "gate and down have incompatible shapes");

    TORCH_CHECK(x.size(1) % 256 == 0,               "x.shape[1] must be multiple of 256");

    TORCH_CHECK(gate_seq_g_idx.device().is_meta() || gate_seq_g_idx.size(0) == gate.size(0) * 2 * 8,    "gate_seq_g_idx and gate have incompatible shapes");
    TORCH_CHECK(gate_x_map.device().is_meta() || gate_x_map.size(0) == gate.size(0) * 8,                "gate_x_map and gate have incompatible shapes");

    TORCH_CHECK(up_seq_g_idx.device().is_meta() || up_seq_g_idx.size(0) == gate.size(0) * 2 * 8,        "up_seq_g_idx and up have incompatible shapes");
    TORCH_CHECK(up_x_map.device().is_meta() || up_x_map.size(0) == gate.size(0) * 8,                    "up_x_map and up have incompatible shapes");

    TORCH_CHECK(down_seq_g_idx.device().is_meta() || down_seq_g_idx.size(0) == gate.size(0) * 2 * 8,    "down_seq_g_idx and down have incompatible shapes");
    TORCH_CHECK(down_x_map.device().is_meta() || down_x_map.size(0) == gate.size(0) * 8,                "down_x_map and down have incompatible shapes");

    int groupsize = gate.size(0) * 8 / gate_zeros.size(0);

    TORCH_CHECK(groupsize * gate_zeros.size(0) == gate.size(0) * 8,     "gate.shape[0] must be a multiple of gate_zeros.shape[0]")
    TORCH_CHECK(groupsize * up_zeros.size(0) == up.size(0) * 8,         "up.shape[0] must be a multiple of up_zeros.shape[0]")
    TORCH_CHECK(groupsize * down_zeros.size(0) == down.size(0) * 8,     "down.shape[0] must be a multiple of down_zeros.shape[0]")

    int height = x.size(0);
    int dim = x.size(1);
    int width = gate.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    _cuda_raise(
        q4v2_mlp_cuda
        (
            (half*) x.data_ptr(),

            (half*) x_temp.data_ptr(),
            (float*) x_col_temp.data_ptr(),
            (half*) x_act_temp.data_ptr(),

            (half*) rms_norm_weight.data_ptr(),
            epsilon,

            (uint32_t*) gate.data_ptr(),
            (half*) gate_scales.data_ptr(),
            (uint32_t*) gate_zeros.data_ptr(),
            gate_seq_g_idx.device().is_meta() ? NULL : (uint16_t*) gate_seq_g_idx.data_ptr(),
            gate_x_map.device().is_meta() ? NULL : (uint32_t*) gate_x_map.data_ptr(),

            (uint32_t*) up.data_ptr(),
            (half*) up_scales.data_ptr(),
            (uint32_t*) up_zeros.data_ptr(),
            up_seq_g_idx.device().is_meta() ? NULL : (uint16_t*) up_seq_g_idx.data_ptr(),
            up_x_map.device().is_meta() ? NULL : (uint32_t*) up_x_map.data_ptr(),

            (uint32_t*) down.data_ptr(),
            (half*) down_scales.data_ptr(),
            (uint32_t*) down_zeros.data_ptr(),
            down_seq_g_idx.device().is_meta() ? NULL : (uint16_t*) down_seq_g_idx.data_ptr(),
            down_x_map.device().is_meta() ? NULL : (uint32_t*) down_x_map.data_ptr(),

            height,
            dim,
            width,
            groupsize
        )
    );
}

void rms_norm
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor scratch,  // shape = (x.shape[0],)
    float epsilon
)
{
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be a half tensor");
    TORCH_CHECK(w.dtype() == torch::kHalf, "w must be a half tensor");
    TORCH_CHECK(scratch.dtype() == torch::kFloat, "scratch must be a float tensor");

    TORCH_CHECK(x.size(1) == w.size(0), "x and w have incompatible shapes");
    TORCH_CHECK(x.size(0) == scratch.size(0), "x and scratch have incompatible shapes");
    TORCH_CHECK(x.size(0) == out.size(0) && x.size(1) == out.size(1), "x and out have incompatible shapes");

    int rows = x.size(0);
    int dim = x.size(1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));

    _cuda_raise(
        rms_norm_cuda
        (
            (half*) x.data_ptr(),
            (half*) w.data_ptr(),
            (half*) out.data_ptr(),
            (float*) scratch.data_ptr(),
            epsilon,
            rows,
            dim
        )
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("q4v2_matmul", &q4v2_matmul, "q4v2 matrix multiplication");
  m.def("q4v2_mlp", &q4v2_mlp, "q4v2 llama mlp");
  m.def("q4v2_recons", &q4v2_recons, "q4v2 matrix reconstruction");
  m.def("q4v2_sequential", &q4v2_sequential, "q4v2 matrix serialization");
  m.def("column_remap", &column_remap, "half matrix column remapping");
  m.def("rms_norm", &rms_norm, "rms norm");
}