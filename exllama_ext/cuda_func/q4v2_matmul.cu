#include "q4v2_matmul.cuh"
#include "column_remap.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#include "../cuda_compat.cuh"
#include "../cuda_buffers.cuh"
#include "half_matmul.cuh"

// Block size

const int THREADS_X = 32;       // Block size and thread count along columns in w and out
const int THREADS_Y = 1;        // Block size and thread count along rows in x and out
const int BLOCK_SIZE_Z = 512;   // Block size (1 thread per block) along columns in x, rows in w

template<bool use_groupsize>
__global__ void q4v2_matmul_kernel
(
    const half* x,
    const uint32_t* w,
    half* out,  // (y)
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize
)
{
    // Start of block

    int x_column = BLOCK_SIZE_Z * blockIdx.z;
    int x_column_end = min(dim, BLOCK_SIZE_Z * (blockIdx.z + 1));

    int w_column = THREADS_X * blockIdx.x + threadIdx.x;
    int x_row = THREADS_Y * blockIdx.y + threadIdx.y;
    int w_row = x_column;

    int iterations = (x_column_end - x_column) / 8;

    // Views

    MatrixView_half x_(x, height, dim);
    MatrixView_half w_scales_(w_scales, dim / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, dim / groupsize, width);
    MatrixView_q4_column w_(w, dim, width);
    MatrixView_half_rw out_(out, height, width);

    // Zero output

    if (blockIdx.z == 0)
    {
        out_.set(x_row, w_column, {});
        __syncthreads();
    }

    // Group for zeros and scales

    int g_idx_idx = x_column * 2;

    // Loop over part of x row (and w column)

    half2 acc = {};

    if constexpr (use_groupsize)
    {
        // For quant matrices where groupsize divides BLOCK_SIZE_Z we always start on a group boundary, so this
        // coule be slightly faster

        for (int k = x_column, group = x_column / groupsize; k < x_column + iterations * 8; group++)
        {
            half2 w_scale = w_scales_.item_half2half2(group, w_column);
            uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

            acc = dot_product_8(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, groupsize / 8);
            k += groupsize;
        }
    }
    else
    {
        // Otherwise assume groupsize is a multiple of 8, do 8 columns per iteration and trust the cache

        for (int k = x_column; k < x_column + iterations * 8; )
        {
            int group = x_column / groupsize;
            half2 w_scale = w_scales_.item_half2half2(group, w_column);
            uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

            acc = dot_product_8(acc, x_, x_row, k, w_, k, w_column, w_scale, w_zero, 1);
            k += 8;
        }
    }

    // Add to block result

    half result = __hadd(acc.x, acc.y);
    atomicAdd(out_.item_ptr(x_row, w_column), result);
}

// Compute y = x @ w
//
// Shape of x is [height, dim], dtype = half
// Shape of w is [dim, width], dtype = q4 (packed columns)
// Output shape is [height, width], dtyle = half
//
// Shape of w_scales is [height / groupsize, width], dtype = q4 (packed rows)
// Shape of w_zeros is [height / groupsize, width], dtype = half

void q4_matmul_cuda
(
    const half* x,
    const int x_height,
    const Q4Matrix* w,
    half* out
)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;

    cudaSetDevice(w->device);

    const half* x_mapped = x;
    if (w->cuda_x_map)
    {
        CudaBuffers* buffers = get_buffers(w->device);
        column_remap_cuda(x, buffers->temp_state, x_height, dim, w->cuda_x_map);
        x_mapped = buffers->temp_state;
    }

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        (dim + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );

    if (BLOCK_SIZE_Z % w->groupsize == 0)
    {
        q4v2_matmul_kernel <true>  <<<blocks, threads>>>(x_mapped, w->cuda_qweight, out, w->cuda_scales, w->cuda_qzeros, height, dim, width, w->groupsize);
    }
    else
    {
        q4v2_matmul_kernel <false> <<<blocks, threads>>>(x_mapped, w->cuda_qweight, out, w->cuda_scales, w->cuda_qzeros, height, dim, width, w->groupsize);
    }
}

void q4_matmul_recons_cuda
(
    const half* x,
    const int x_height,
    Q4Matrix* w,
    half* out,
    const cublasHandle_t handle
)
{
    int height = x_height;
    int dim = w->height;
    int width = w->width;

    cudaSetDevice(w->device);
    CudaBuffers* buffers = get_buffers(w->device);

    const half* x_mapped = x;
    if (w->cuda_x_map)
    {
        column_remap_cuda(x, buffers->temp_state, x_height, dim, w->cuda_x_map);
        x_mapped = buffers->temp_state;
    }

    w->reconstruct(buffers->temp_dq);

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, buffers->temp_dq, width, x_mapped, dim, &beta, out, width);
}