#include "q4v2_matmul.cuh"
#include "column_remap.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#include "../cuda_compat.cuh"

// Block size

const int THREADS_X = 32;       // Block size and thread count along columns in w and out
const int THREADS_Y = 1;        // Block size and thread count along rows in x and out
const int BLOCK_SIZE_Z = 512;   // Block size (1 thread per block) along columns in x, rows in w

template<bool use_g_idx, bool use_groupsize>
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
    const int groupsize,
    const uint16_t* seq_g_idx
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

    if constexpr (use_g_idx)
    {
        // Group index version

        for (int k = 0; k < iterations; k++)
        {
            int group_rem = seq_g_idx[g_idx_idx + 1];
            if (group_rem >= 8)
            {
                // Go faster if next 8 group indices are the same

                int group = seq_g_idx[g_idx_idx];
                int group_len = seq_g_idx[g_idx_idx + 1];
                int group_len_8 = group_len / 8;
                g_idx_idx += group_len_8 * 8 * 2;

                half2 w_scale = w_scales_.item_half2half2(group, w_column);
                uint32_t w_zero = w_zeros_.item(group, w_column) + 1;

                acc = dot_product_8(acc, x_, x_row, x_column + k * 8, w_, w_row + k * 8, w_column, w_scale, w_zero, group_len_8);

            }
            else
            {
                #pragma unroll
                for (int l = 0; l < 8; l += 2)
                {
                    // Reconstruct two values from w with independent group indices

                    int group_l = seq_g_idx[g_idx_idx]; seq_g_idx += 2;
                    int group_r = seq_g_idx[g_idx_idx]; seq_g_idx += 2;
                    int w_zero_l = w_zeros_.item(group_l, w_column) + 1;
                    int w_zero_r = w_zeros_.item(group_r, w_column) + 1;
                    half2 w_scale = __halves2half2(w_scales_.item(group_l, w_column), w_scales_.item(group_r, w_column));
                    half w_0 = __int2half_rn(w_.item(w_row + k * 8 + l + 0, w_column) - w_zero_l);
                    half w_1 = __int2half_rn(w_.item(w_row + k * 8 + l + 1, w_column) - w_zero_r);
                    half2 w_01 = __halves2half2(w_0, w_1);
                    w_01 = __hmul2(w_01, w_scale);

                    half2 x_01 = x_.item_half2(x_row, x_column + k * 8 + l + 0);
                    acc = __hfma2(x_01, w_01, acc);
                }
            }
        }
    }
    else
    {
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

cudaError_t q4v2_matmul_cuda
(
    const half* x,
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize,
    const uint16_t* seq_g_idx,
    const uint32_t* x_map
)
{
    cudaError_t _cuda_err = cudaSuccess;

    // Temp buffers

    half* x_mapped = NULL;

    // Remap x if x_map given

    if (x_map)
    {
        _cuda_check(cudaMalloc(&x_mapped, height * dim * sizeof(half)));
        _cuda_check(column_remap_cuda(x, x_mapped, height, dim, x_map));

//         cudaDeviceSynchronize();
//         _cuda_check(cudaGetLastError());
    }

    // Multiply

    {
        dim3 threads(THREADS_X, THREADS_Y, 1);

        dim3 blocks
        (
            (width + threads.x - 1) / threads.x,
            (height + threads.y - 1) / threads.y,
            (dim + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
        );

        // TODO: Clean this all up a bit at some point

        if (BLOCK_SIZE_Z % groupsize == 0)
        {
            if (seq_g_idx) q4v2_matmul_kernel <true,  true>  <<<blocks, threads>>>(x_map ? x_mapped : x, w, out, w_scales, w_zeros, height, dim, width, groupsize, seq_g_idx);
            else           q4v2_matmul_kernel <false, true>  <<<blocks, threads>>>(x_map ? x_mapped : x, w, out, w_scales, w_zeros, height, dim, width, groupsize, seq_g_idx);
        }
        else
        {
            if (seq_g_idx) q4v2_matmul_kernel <true,  false> <<<blocks, threads>>>(x_map ? x_mapped : x, w, out, w_scales, w_zeros, height, dim, width, groupsize, seq_g_idx);
            else           q4v2_matmul_kernel <false, false> <<<blocks, threads>>>(x_map ? x_mapped : x, w, out, w_scales, w_zeros, height, dim, width, groupsize, seq_g_idx);
        }

//         cudaDeviceSynchronize();
//         _cuda_check(cudaGetLastError());
    }

    // Clean up

_cuda_fail:

    if (x_mapped) cudaFree(x_mapped);

    return _cuda_err;
}
