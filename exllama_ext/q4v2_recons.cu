#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// Block size

const int THREADS_X = 64;     // Block size and thread count along columns in out, each thread converts 2 columns
const int THREADS_Y = 4;       // Block size and thread count along rows in x and out, each thread converts BLOCK_SIZE_Y * 8 rows
const int BLOCK_SIZE_Y = 1;    // * 8 rows

__global__ void q4v2_recons_kernel
(
    const uint32_t* w,
    half* out,  // (y)
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int width,
    const int groupsize
)
{
    // Start of input

    uint64_t* w2 = (uint64_t*) w;
    int w2_column = THREADS_X * blockIdx.x + threadIdx.x;
    int w2_row = (THREADS_Y * blockIdx.y + threadIdx.y) * BLOCK_SIZE_Y;
    int w2_stride = width >> 1;

    // Start of output

    half2* out2 = (half2*) out;
    int out2_column = w2_column;
    int out2_row = w2_row * 8;
    int out2_stride = width >> 1;

    // Group, zeros and scales

    int groupsize8 = groupsize >> 3;
    int group_idx = w2_row / groupsize8;
    int next_group = group_idx * groupsize8;  // first iteration will advance to first group

    half2* w_scales2 = (half2*) w_scales;
    int w_scales2_column = w2_column;
    int w_scales2_stride = width >> 1;

    uint64_t* w_zeros2 = (uint64_t*) w_zeros;
    int w_zeros2_column = w2_column >> 3;           // w_column >> 4
    int w_zeros2_shift = (w2_column & 0x07) << 3;   // shift 2*4 bits per double column
    int w_zeros2_stride = (width >> 4);

    // Indices

    int w2_idx = w2_row * w2_stride + w2_column;
    int w_scales2_idx = group_idx * w_scales2_stride + w_scales2_column;
    int w_zeros2_idx = group_idx * w_zeros2_stride + w_zeros2_column;

    int out2_idx = out2_row * out2_stride + out2_column;

    // Loop over BLOCK_SIZE_Y

    //int w2_row_end = min(w2_row + BLOCK_SIZE_Y, height);
    int w2_row_end = w2_row + BLOCK_SIZE_Y;

    half2 w_scale2;
    int w_zerol_q, w_zeror_q;

    while (w2_row < w2_row_end)
    {
        if (BLOCK_SIZE_Y == 1 || w2_row >= next_group)  // optimizer should remove this if BLOCK_SIZE_Y == 1
        {
            w_scale2 = w_scales2[w_scales2_idx];

            uint64_t w_zero2_packed = w_zeros2[w_zeros2_idx];
            int w_zero2_q = ((w_zero2_packed >> w_zeros2_shift) & 0xff) + 0x11;
            w_zerol_q = w_zero2_q & 0x0f;
            w_zeror_q = w_zero2_q >> 4;

            w_scales2_idx += w_scales2_stride;
            w_zeros2_idx += w_zeros2_stride;
            next_group += groupsize8;
        }

        // Read 2 * 8 packed quants from w2

        uint64_t w2_read = w2[w2_idx];
        w2_idx += w2_stride;

        // Convert quants to half2

        half2 w2_0 = __halves2half2(__int2half_rn((int)((w2_read >>  0) & 0x0f) - w_zerol_q), __int2half_rn((int)((w2_read >> 32) & 0x0f) - w_zeror_q));
        half2 w2_1 = __halves2half2(__int2half_rn((int)((w2_read >>  4) & 0x0f) - w_zerol_q), __int2half_rn((int)((w2_read >> 36) & 0x0f) - w_zeror_q));
        half2 w2_2 = __halves2half2(__int2half_rn((int)((w2_read >>  8) & 0x0f) - w_zerol_q), __int2half_rn((int)((w2_read >> 40) & 0x0f) - w_zeror_q));
        half2 w2_3 = __halves2half2(__int2half_rn((int)((w2_read >> 12) & 0x0f) - w_zerol_q), __int2half_rn((int)((w2_read >> 44) & 0x0f) - w_zeror_q));
        half2 w2_4 = __halves2half2(__int2half_rn((int)((w2_read >> 16) & 0x0f) - w_zerol_q), __int2half_rn((int)((w2_read >> 48) & 0x0f) - w_zeror_q));
        half2 w2_5 = __halves2half2(__int2half_rn((int)((w2_read >> 20) & 0x0f) - w_zerol_q), __int2half_rn((int)((w2_read >> 52) & 0x0f) - w_zeror_q));
        half2 w2_6 = __halves2half2(__int2half_rn((int)((w2_read >> 24) & 0x0f) - w_zerol_q), __int2half_rn((int)((w2_read >> 56) & 0x0f) - w_zeror_q));
        half2 w2_7 = __halves2half2(__int2half_rn((int)((w2_read >> 28) & 0x0f) - w_zerol_q), __int2half_rn((int)((w2_read >> 60) & 0x0f) - w_zeror_q));

        out2[out2_idx] = __hmul2(w2_0, w_scale2); out2_idx += out2_stride;
        out2[out2_idx] = __hmul2(w2_1, w_scale2); out2_idx += out2_stride;
        out2[out2_idx] = __hmul2(w2_2, w_scale2); out2_idx += out2_stride;
        out2[out2_idx] = __hmul2(w2_3, w_scale2); out2_idx += out2_stride;
        out2[out2_idx] = __hmul2(w2_4, w_scale2); out2_idx += out2_stride;
        out2[out2_idx] = __hmul2(w2_5, w_scale2); out2_idx += out2_stride;
        out2[out2_idx] = __hmul2(w2_6, w_scale2); out2_idx += out2_stride;
        out2[out2_idx] = __hmul2(w2_7, w_scale2); out2_idx += out2_stride;

        w2_row++;
    }
}


// Convert w -> y, from q4 to half
//
// Shape of w is [height, width], dtype = q4
// Output shape is [height, width], dtyle = half
// Shape of w_scales is [height / groupsize, width], dtype = 4-bit quant (packed rows)
// Shape of w_zeros is [height / groupsize, width], dtype = half

void q4v2_recons_cuda
(
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
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
        (width + threads.x - 1) / threads.x / 2,
        (height + threads.y - 1) / threads.y / BLOCK_SIZE_Y,
        1
    );

    q4v2_recons_kernel<<<blocks, threads>>>
    (
        w,
        out,
        w_scales,
        w_zeros,
        height,
        width,
        groupsize
    );
}

