#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Block size

const int THREADS_X = 32;       // Block size and thread count along columns in w and out
const int THREADS_Y = 1;        // Block size and thread count along rows in x and out
const int BLOCK_SIZE_Z = 256;   // Block size (1 thread per block) along columns in x, rows in w

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
    int w_column = THREADS_X * blockIdx.x + threadIdx.x;
    int x_row = THREADS_Y * blockIdx.y + threadIdx.y;

    int x_stride = dim;

    // Row index in w, which packs quants vertically, i.e.:
    //
    // w[width * 0 + 0] = little_endian_packed(r0c0, r1c0,  r2c0,  r3c0,  r4c0,  r5c0,  r6c0,  r7c0)
    // w[width * 1 + 0] = little_endian_packed(r8c0, r9c0, r10c0, r11c0, r12c0, r13c0, r14c0, r15c0)
    // w[width * 0 + 1] = little_endian_packed(r0c1, r1c1,  r2c1,  r3c1,  r4c1,  r5c1,  r6c1,  r7c1)
    // w[width * 1 + 1] = little_endian_packed(r8c1, r9c1, r10c1, r11c1, r12c1, r13c1, r14c1, r15c1)

    int w_row = x_column >> 3;
    int w_stride = width;

    // Group for zeros and scales

    int group_idx = x_column / groupsize;
    int next_group = group_idx * groupsize;  // first iteration will advance to first group

    // w_scales is half types in groups:
    //
    // w_scales[width * 0 + 4] = r[0             .. 1 * groupsize] c4
    // w_scales[width * 1 + 4] = r[1 * groupsize .. 2 * groupsize] c4
    // w_scales[width * 2 + 4] = r[2 * groupsize .. 3 * groupsize] c4
    //
    // w_scales_row = group_idx;

    int w_scales_column = w_column;
    int w_scales_stride = width;

    // w_zeros packs zeros horizontally, groups vertically:
    //
    // w_zeros[width/8 * 0 + 0] = little_endian_packed(r0c0, r0c1,  r0c2,  r0c3,  r0c4,  r0c5,  r0c6,  r0c7)
    // w_zeros[width/8 * 1 + 1] = little_endian_packed(rgc8, rgc9, rgc10, rgc11, rgc12, rgc13, rgc14, rgc15) where g = groupsize
    //
    // w_zeros_row = group_idx;

    int w_zeros_column = w_column >> 3;
    int w_zeros_shift = (w_column & 0x07) << 2;
    int w_zeros_stride = (width >> 3);

    // Indices

    int x_idx = x_row * x_stride + x_column;
    int w_idx = w_row * w_stride + w_column;
    int w_scales_idx = group_idx * w_scales_stride + w_scales_column;
    int w_zeros_idx = group_idx * w_zeros_stride + w_zeros_column;

    int out_idx = x_row * width + w_column;

    // Loop over part of x row (and w column)

    int x_column_end = x_column + BLOCK_SIZE_Z;
    half2 acc = {};
    half2 w_scale;
    int w_zero_q;

    while (x_column < x_column_end)
    {
        // Only extract scale and zero at group boundary

        if (x_column >= next_group)
        {
            w_scale = __half2half2(w_scales[w_scales_idx]);

            uint32_t w_zero_packed = w_zeros[w_zeros_idx];
            w_zero_q = ((w_zero_packed >> w_zeros_shift) & 0x0f) + 1;

            w_scales_idx += w_scales_stride;
            w_zeros_idx += w_zeros_stride;
            next_group += groupsize;
        }

        // Read 8 packed quants from w

        uint32_t w_read = w[w_idx];
        w_idx += w_stride;

        // Convert quants to half2

        half w_0 = __int2half_rn((int)((w_read) & 0x0f) - w_zero_q); w_read >>= 4;
        half w_1 = __int2half_rn((int)((w_read) & 0x0f) - w_zero_q); w_read >>= 4;
        half w_2 = __int2half_rn((int)((w_read) & 0x0f) - w_zero_q); w_read >>= 4;
        half w_3 = __int2half_rn((int)((w_read) & 0x0f) - w_zero_q); w_read >>= 4;
        half w_4 = __int2half_rn((int)((w_read) & 0x0f) - w_zero_q); w_read >>= 4;
        half w_5 = __int2half_rn((int)((w_read) & 0x0f) - w_zero_q); w_read >>= 4;
        half w_6 = __int2half_rn((int)((w_read) & 0x0f) - w_zero_q); w_read >>= 4;
        half w_7 = __int2half_rn((int)((w_read) & 0x0f) - w_zero_q);

        half2 w_01 = __halves2half2(w_0, w_1);
        half2 w_23 = __halves2half2(w_2, w_3);
        half2 w_45 = __halves2half2(w_4, w_5);
        half2 w_67 = __halves2half2(w_6, w_7);

        w_01 = __hmul2(w_01, w_scale);
        w_23 = __hmul2(w_23, w_scale);
        w_45 = __hmul2(w_45, w_scale);
        w_67 = __hmul2(w_67, w_scale);

        // Read 8 halves from x

        half2* x_h2 = (half2*) (x + x_idx);
        x_idx += 8;

        half2 x_01 = x_h2[0];
        half2 x_23 = x_h2[1];
        half2 x_45 = x_h2[2];
        half2 x_67 = x_h2[3];

        // Multiply and accumulate

        acc = __hfma2(x_01, w_01, acc);
        acc = __hfma2(x_23, w_23, acc);
        acc = __hfma2(x_45, w_45, acc);
        acc = __hfma2(x_67, w_67, acc);

        x_column += 8;
    }

    // Add to block result

    half result = __hadd(acc.x, acc.y);
    atomicAdd(&out[out_idx], result);
    //out[out_idx] = result;
}


// Compute y = x @ w
//
// Shape of x is [height, dim], dtype = half
// Shape of w is [dim, width], dtype = q4 (packed columns)
// Output shape is [height, width], dtyle = half
//
// Shape of w_scales is [height / groupsize, width], dtype = q4 (packed col)
// Shape of w_zeros is [height / groupsize, width], dtype = half

void q4v2_matmul_cuda
(
    const half* x,
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize
)
{
    dim3 threads
    (
        THREADS_X,
        THREADS_Y,
        1
    );

    dim3 blocks
    (
        (width + threads.x - 1) / threads.x,
        (height + threads.y - 1) / threads.y,
        (dim + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z
    );

    q4v2_matmul_kernel<<<blocks, threads>>>
    (
        x,
        w,
        out,
        w_scales,
        w_zeros,
        height,
        dim,
        width,
        groupsize
    );
}
