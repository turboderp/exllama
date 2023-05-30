#include "q4v2_recons.cuh"
#include "../util.cuh"
#include "../matrix.cuh"

// Block size

const int THREADS_X = 64;      // Block size and thread count along columns in out, each thread converts 1 column
const int THREADS_Y = 1;       // Block size and thread count along rows in x and out, each thread converts 8 rows

template<bool use_g_idx>
__global__ void q4v2_recons_kernel
(
    const uint32_t* w,
    half* out,  // (y)
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int width,
    const int groupsize,
    const uint16_t* seq_g_idx
)
{
    // Start of block

    int column = THREADS_X * blockIdx.x + threadIdx.x;
    int row = (THREADS_Y * blockIdx.y + threadIdx.y) * 8;

    // Views

    MatrixView_q4_column w_(w, height, width);
    MatrixView_half_rw out_(out, height, width);
    MatrixView_half w_scales_(w_scales, height / groupsize, width);
    MatrixView_q4_row w_zeros_(w_zeros, height / groupsize, width);

    if constexpr (use_g_idx)
    {
        // Group index version

        uint32_t w_read = w_.item_uint32_t(row, column);
        half* out_ptr = out_.item_ptr(row, column);
        int rows_left = 8;

        while (rows_left > 0)
        {
            int g_idx_idx = (row / groupsize) * 2;
            int group = seq_g_idx[g_idx_idx];
            int group_rows = min(seq_g_idx[g_idx_idx + 1], rows_left);
            g_idx_idx += 2;

            half w_scale = w_scales_.item(group, column);
            uint32_t w_zero = w_zeros_.item(group, column) + 1;

            for (; group_rows > 0; group_rows--, rows_left--)
            {
                half w_item = __hmul(__int2half_rn((int)(w_read & 0x0f) - w_zero), w_scale);
                *out_ptr = w_item; out_ptr += out_.width;
            }
        }
    }
    else
    {
        // Groupsize version

        int group = row / groupsize;

        half w_scale = w_scales_.item(group, column);
        uint32_t w_zero = w_zeros_.item(group, column) + 1;

        uint32_t w_read = w_.item_uint32_t(row, column);
        half* out_ptr = out_.item_ptr(row, column);

        #pragma unroll
        for (int s = 0; s < 32; s += 4)
        {
            half w_item = __hmul(__int2half_rn((int)((w_read >> s) & 0x0f) - w_zero), w_scale);
            *out_ptr = w_item; out_ptr += out_.width;
        }
    }
}

// Convert w -> y, from q4 to half
//
// Shape of w is [height, width], dtype = q4
// Output shape is [height, width], dtyle = half
// Shape of w_scales is [height / groupsize, width], dtype = 4-bit quant (packed rows)
// Shape of w_zeros is [height / groupsize, width], dtype = half

cudaError_t q4v2_recons_cuda
(
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int width,
    const int groupsize,
    const uint16_t* seq_g_idx
)
{
    cudaError_t _cuda_err = cudaSuccess;

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        1
    );

    if (seq_g_idx) q4v2_recons_kernel <true>  <<<blocks, threads>>>(w, out, w_scales, w_zeros, height, width, groupsize, seq_g_idx);
    else           q4v2_recons_kernel <false> <<<blocks, threads>>>(w, out, w_scales, w_zeros, height, width, groupsize, seq_g_idx);

//     cudaDeviceSynchronize();
//     _cuda_check(cudaGetLastError());
//
// _cuda_fail:

    return _cuda_err;
}