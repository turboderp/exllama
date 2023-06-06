#include "half_matmul.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#include "../cuda_compat.cuh"
#if defined(USE_ROCM)
#include "../hip_compat.cuh"
#endif

// Block size

const int THREADS_X = 32;     // Block size and thread count along columns in w and out
const int THREADS_Y = 8;      // Block size and thread count along rows in x and out
const int BLOCKSIZE = 256;

__global__ void half_matmul_kernel
(
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width
)
{
    const int column = (blockIdx.x * THREADS_X + threadIdx.x) * 2;
    const int row = blockIdx.y * THREADS_Y + threadIdx.y;
    const int k0 = blockIdx.z * BLOCKSIZE;

    if (row >= height) return;
    if (column >= width) return;

    MatrixView_half x_(x, height, dim);
    MatrixView_half w_(w, dim, width);
    MatrixView_half_rw out_(out, height, width);

    half2* x_ptr = (half2*) x_.item_ptr(row, k0);
    half2* w_ptr = (half2*) w_.item_ptr(k0, column);
    half2 acc = {};

    #pragma unroll
    for (int k = k0; k < k0 + BLOCKSIZE / 2; k++)
    {
        half2 x_item = *x_ptr++;
        half2 x_item_0 = __half2half2(x_item.x);
        half2 x_item_1 = __half2half2(x_item.y);
        half2 w_item_0 = *w_ptr; w_ptr += w_.width / 2;
        half2 w_item_1 = *w_ptr; w_ptr += w_.width / 2;
        acc = __hfma2(x_item_0, w_item_0, acc);
        acc = __hfma2(x_item_1, w_item_1, acc);
    }

    // out_.set(row, column, acc);
    atomicAdd((half2*)out_.item_ptr(row, column), acc);
}

void half_matmul_cuda
(
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width
)
{
    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (width + THREADS_X - 1) / THREADS_X / 2,
        (height + THREADS_Y - 1) / THREADS_Y,
        (dim + BLOCKSIZE - 1) / BLOCKSIZE
    );

    half_matmul_kernel<<<blocks, threads>>>(x, w, out, height, dim, width);
}

// cuBLAS can't be beat for large matrices, probably

void half_matmul_cublas_cuda
(
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width,
    cublasHandle_t handle
)
{
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);

    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, w, width, x, dim, &beta, out, width);
}
