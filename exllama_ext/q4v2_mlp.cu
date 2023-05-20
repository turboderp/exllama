#include "q4v2_mlp.h"
#include "util.h"
#include "matrix.h"
#include "q4v2_matmul.h"
#include "rms_norm.h"

const int THREADS_X = 32;
const int THREADS_Y = 4;
// const int MAX_DIMENSION = 8192;

__device__ __forceinline__ half silu(half x)
{
    half one = __float2half(1.0f);
    half neg_x = __hneg(x);
    half e = hexp(neg_x);
    half sum = __hadd(one, e);
    half r = hrcp(sum);
    half result = __hmul(x, r);
    return result;
}

__global__ void q4v2_mlp_kernel_slow
(
    const half* x,
    half* out,

    const uint32_t* gate,
    const half* gate_scales,
    const uint32_t* gate_zeros,
    const uint16_t* gate_seq_g_idx,
    const uint32_t* gate_x_map,

    const uint32_t* up,
    const half* up_scales,
    const uint32_t* up_zeros,
    const uint16_t* up_seq_g_idx,
    const uint32_t* up_x_map,

    const int height,
    const int dim,
    const int width,
    const int groupsize
)
{
    // Start of block

    int column = THREADS_X * blockIdx.x + threadIdx.x;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= height) return;

    // Views

    MatrixView_half x_(x, height, dim);
    MatrixView_half_rw out_(out, height, width);

    MatrixView_q4_column gate_(gate, dim, width);
    MatrixView_q4_row gate_zeros_(gate_zeros, dim / groupsize, width);
    MatrixView_half gate_scales_(gate_scales, dim / groupsize, width);

    MatrixView_q4_column up_(up, dim, width);
    MatrixView_q4_row up_zeros_(up_zeros, dim / groupsize, width);
    MatrixView_half up_scales_(up_scales, dim / groupsize, width);

    // Preload row in x

//     __shared__ half x_row_buffer[MAX_DIMENSION];
//     if (threadIdx.x == 0)
//     {
//         const uint64_t* a = (const uint64_t*) x_.item_ptr(row, 0);
//         uint64_t* b = (uint64_t*) x_row_buffer;
//         for (int i = 0; i < dim / 4; i++) *b++ = *a++;
//     }
//     __syncthreads();

    // Dot products of whole row in x with whole column in gate and up

    half2 acc = {};  // (gate, up)

    for (int k = 0, group = 0; k < dim; group++, k += groupsize)
    {
        half2 gate_scale = gate_scales_.item_half2half2(group, column);
        uint32_t gate_zero = gate_zeros_.item(group, column) + 1;
        half2 up_scale = up_scales_.item_half2half2(group, column);
        uint32_t up_zero = up_zeros_.item(group, column) + 1;

        //acc = dot_product_8_dual_buffered(acc, x_row_buffer, row, k, gate_, up_, k, column, gate_scale, gate_zero, up_scale, up_zero, groupsize / 8);
        acc = dot_product_8_dual(acc, x_, row, k, gate_, up_, k, column, gate_scale, gate_zero, up_scale, up_zero, groupsize / 8);
    }

    half gate_dot = acc.x;
    half up_dot = acc.y;

    // SiLU activation and gate

    half result = __hmul(silu(gate_dot), up_dot);

    // Store (SiLU(x @ gate_proj) * (x @ up_proj)) [row, column]

    out_.set(row, column, result);
}


cudaError_t q4v2_mlp_cuda
(
    half* x,                        // shape == (height, dim)

    half* x_temp,                   // shape == x.shape
    float* x_col_temp,              // shape == (x.shape[0],) == (height,)
    half* x_act_temp,               // shape == (x.shape[0], gate.shape[1]) == (height, width)

    const half* rms_norm_weight,    // shape == (x.shape[1],) == (dim,)
    float epsilon,

    const uint32_t* gate,           // shape == (dim, width)
    const half* gate_scales,
    const uint32_t* gate_zeros,
    const uint16_t* gate_seq_g_idx,
    const uint32_t* gate_x_map,

    const uint32_t* up,
    const half* up_scales,
    const uint32_t* up_zeros,
    const uint16_t* up_seq_g_idx,
    const uint32_t* up_x_map,

    const uint32_t* down,
    const half* down_scales,
    const uint32_t* down_zeros,
    const uint16_t* down_seq_g_idx,
    const uint32_t* down_x_map,

    const int height,
    const int dim,
    const int width,
    const int groupsize
)
{
    cudaError_t _cuda_err = cudaSuccess;

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (width + THREADS_X - 1) / THREADS_X,
        (height + THREADS_Y - 1) / THREADS_Y,
        1
    );

    // x_temp = rms_layernorm(x)

    cudaMemset(x_col_temp, 0, height * sizeof(float));

    _cuda_err = rms_norm_cuda(x, rms_norm_weight, x_temp, x_col_temp, epsilon, height, dim);
    if (_cuda_err != cudaSuccess) goto _cuda_fail;

    // x_act_temp = silu(x_temp @ gate_proj) * (x_temp @ up_proj)

    q4v2_mlp_kernel_slow<<<blocks, threads>>>
    (
        x_temp,
        x_act_temp,

        gate,
        gate_scales,
        gate_zeros,
        gate_seq_g_idx,
        gate_x_map,

        up,
        up_scales,
        up_zeros,
        up_seq_g_idx,
        up_x_map,

        height,
        dim,
        width,
        groupsize
    );

    // x += x_act_temp @ down_proj

    _cuda_err = q4v2_matmul_cuda(x_act_temp, down, x, down_scales, down_zeros, height, width, dim, groupsize, gate_seq_g_idx, NULL);
    if (_cuda_err != cudaSuccess) goto _cuda_fail;

_cuda_fail:

    return _cuda_err;
}