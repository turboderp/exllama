#include "column_remap.cuh"
#include "../util.cuh"

// Using 1024 make me crash with "Memory access fault by GPU node-1 (Agent
// handle: 0x012345678912) on address 0x012345678912. Reason: Page not present
// or supervisor privilege."
#if defined(USE_ROCM)
const int SHUF_BLOCKSIZE_X = 256;
#else
const int SHUF_BLOCKSIZE_X = 1024;
#endif
const int SHUF_BLOCKSIZE_Y = 16;

__global__ void column_remap_kernel
(
    const half* x,
    half* x_new,
    const int x_width,
    const int x_height,
    const uint32_t* x_map
)
{
    int x_column = SHUF_BLOCKSIZE_X * blockIdx.x + threadIdx.x;
    int x_row = SHUF_BLOCKSIZE_Y * blockIdx.y;

    int x_stride = x_width;
    int x_idx = x_row * x_stride + x_column;

    int x_row_end = min(x_row + SHUF_BLOCKSIZE_Y, x_height);
    int x_idx_end = x_row_end * x_stride + x_column;

    int s_column = x_map[x_column];
    int s_idx = x_row * x_stride + s_column;

    while (x_idx < x_idx_end)
    {
        x_new[x_idx] = x[s_idx];
        x_idx += x_stride;
        s_idx += x_stride;
    }
}


// Remap columns in x to correspond to sequential group index before matmul
//
// perform x -> seq_x such that seq_x @ seq_w == x @ w

void column_remap_cuda
(
    const half* x,
    half* x_new,
    const int x_height,
    const int x_width,
    const uint32_t* x_map
)
{
    dim3 threads(SHUF_BLOCKSIZE_X, 1, 1);

    dim3 blocks
    (
        (x_width + SHUF_BLOCKSIZE_X - 1) / SHUF_BLOCKSIZE_X,
        (x_height + SHUF_BLOCKSIZE_Y - 1) / SHUF_BLOCKSIZE_Y,
        1
    );

    column_remap_kernel<<<blocks, threads>>>(x, x_new, x_width, x_height, x_map);
}
