#include "q4v2_mlp.cuh"
#include "q4v2_matmul.cuh"
#include "rms_norm.cuh"
#include "../cuda_buffers.cuh"
#include "../util.cuh"
#include "../matrix.cuh"

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

__device__ __forceinline__ half2 silu(half2 x)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    half2 r = h2rcp(sum);
    half2 result = __hmul2(x, r);
    return result;
}

__global__ void silu_mul_cuda_kernel
(
    half* x,
    const half* y,
    const int height,
    const int width
)
{
    MatrixView_half_rw x_(x, height, width);
    MatrixView_half y_(y, height, width);

    int column = (THREADS_X * blockIdx.x + threadIdx.x) * 2;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= height) return;

    // silu(x) * y

    half2 one = __half2half2(__float2half(1.0f));

    half2 x_item = x_.item_half2(row, column);
    half2 y_item = y_.item_half2(row, column);

    x_item = silu(x_item);
    x_item = __hmul2(x_item, y_item);

    x_.set_half2(row, column, x_item);
}

// __global__ void half_add_cuda_kernel
// (
//     half* x,
//     const half* y,
//     const int height,
//     const int width
// )
// {
//     MatrixView_half_rw x_(x, height, width);
//     MatrixView_half y_(y, height, width);
//
//     int column = (THREADS_X * blockIdx.x + threadIdx.x) * 2;
//     int row = THREADS_Y * blockIdx.y + threadIdx.y;
//     if (row >= height) return;
//
//     half2 x_item = x_.item_half2(row, column);
//     half2 y_item = y_.item_half2(row, column);
//     half2 sum_item = __hadd2(x_item, y_item);
//     x_.set_half2(row, column, x_item);
// }

cudaError_t q4v2_mlp_cuda
(
    half* x,                        // shape == (height, dim)
    half* out,                      // shape == x.shape

    const half* rms_norm_weight,    // shape == (x.shape[1],) == (dim,)
    float epsilon,

    const uint32_t* gate,           // shape == (dim, width)
    const half* gate_scales,
    const uint32_t* gate_zeros,
    const uint16_t* gate_seq_g_idx,
    const uint32_t* gate_x_map,
    const int gate_groupsize,

    const uint32_t* up,
    const half* up_scales,
    const uint32_t* up_zeros,
    const uint16_t* up_seq_g_idx,
    const uint32_t* up_x_map,
    const int up_groupsize,

    const uint32_t* down,
    const half* down_scales,
    const uint32_t* down_zeros,
    const uint16_t* down_seq_g_idx,
    const uint32_t* down_x_map,
    const int down_groupsize,

    const int height,
    const int dim,
    const int width,

    const int device_index
)
{
    cudaError_t _cuda_err = cudaSuccess;

    CudaBuffers* buffers = get_buffers(device_index);
    buffers->zero_mlp_temp(height);

    // x_temp(out) = rms_layernorm(x)

    rms_norm_cuda(x, rms_norm_weight, out, epsilon, height, dim, device_index);

    // temp1 = x_temp(out) @ gate
    // temp2 = x_temp(out) @ up

    q4v2_matmul_cuda(out, gate, buffers->mlp_temp, gate_scales, gate_zeros, height, dim, width, gate_groupsize, gate_seq_g_idx, gate_x_map);
    q4v2_matmul_cuda(out, up, buffers->mlp_temp + buffers->intermediate_size * height, up_scales, up_zeros, height, dim, width, up_groupsize, up_seq_g_idx, up_x_map);

    // temp1 = silu(temp1) * temp2

    {
        dim3 threads(THREADS_X, THREADS_Y, 1);

        dim3 blocks
        (
            (width + THREADS_X - 1) / THREADS_X / 2,
            (height + THREADS_Y - 1) / THREADS_Y,
            1
        );

        silu_mul_cuda_kernel<<<blocks, threads>>>
        (
            buffers->mlp_temp,
            buffers->mlp_temp + buffers->intermediate_size * height,
            height,
            width
        );
    }

    // x = temp1 @ down

    q4v2_matmul_cuda(buffers->mlp_temp, down, out, down_scales, down_zeros, height, width, dim, down_groupsize, down_seq_g_idx, down_x_map);

//     {
//         dim3 threads(THREADS_X, THREADS_Y, 1);
//
//         dim3 blocks
//         (
//             (buffers->hidden_size + THREADS_X - 1) / THREADS_X / 2,
//             (height + THREADS_Y - 1) / THREADS_Y,
//             1
//         );
//
//         half_add_cuda_kernel<<<blocks, threads>>>
//         (
//             x,
//             buffers->state_temp,
//             height,
//             buffers->hidden_size
//         );
//     }

//_cuda_fail:

    return _cuda_err;
}