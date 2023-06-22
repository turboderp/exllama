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
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out,
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
    const int width,
    cudaStream_t alt_stream
)
{
    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (width + THREADS_X - 1) / THREADS_X / 2,
        (height + THREADS_Y - 1) / THREADS_Y,
        (dim + BLOCKSIZE - 1) / BLOCKSIZE
    );

    half_matmul_kernel<<<blocks, threads, 0, alt_stream>>>(x, w, out, height, dim, width);
}

// cuBLAS can't be beat for large matrices, probably

const int MAX_DIM_SMALL = 8192;

void half_matmul_cublas_cuda
(
    ExLlamaTuning* tuningParams,
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width,
    cublasHandle_t handle,
    bool no_zero,
    cudaStream_t alt_stream
)
{
    // Fall back on a naive kernel for small matmuls to avoid cuBLAS overhead

    if (height < 4 && dim <= MAX_DIM_SMALL)
    {
        half_matmul_small_cuda(tuningParams, x, w, out, height, dim, width, no_zero, alt_stream);
        return;
    }

    // printf("cuBLAS: (%i, %i) @ (%i, %i) -> (%i, %i)\n", height, dim, dim, width, height, width);

    // Use cuBLAS

    const half alpha = __float2half(1.0f);
    const half beta = no_zero ? __float2half(1.0f) : __float2half(0.0f);

    cudaStream_t default_stream;
    if (alt_stream)
    {
        cublasGetStream(handle, &default_stream);
        cublasSetStream(handle, alt_stream);
    }

    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, width, height, dim, &alpha, w, width, x, dim, &beta, out, width);

    if (alt_stream)
    {
        cublasSetStream(handle, default_stream);
    }
}

// Alternative to cuBLAS for tall or wide matrices

const int S_THREADS_X = 8;                                      // width
const int S_THREADS_Z = 1;                                      // height
const int S_BLOCKSIZE = MAX_DIM_SMALL / 1024 * S_THREADS_X;     // dim

template<bool use_half2, bool odd_rank>
__global__ void half_matmul_small_kernel
(
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out,
    const int height,
    const int dim,
    const int width,
    bool no_zero
)
{
    int column = blockIdx.x * S_THREADS_X + threadIdx.x;
    int row = blockIdx.z * S_THREADS_Z + threadIdx.z;
    int k = threadIdx.y * S_BLOCKSIZE;

    if (row >= height) return;
    if (column >= width) return;
    // if (k >= dim) return;
    // printf("%i, %i, %i\n", row, column, k);

    MatrixView_half x_(x, height, dim);
    MatrixView_half w_(w, dim, width);
    MatrixView_half_rw out_(out, height, width);

    int k_end = k + S_BLOCKSIZE;
    if (k_end > dim) k_end = dim;

    const half* x_ptr = x_.item_ptr(row, k);
    const half* x_ptr_end = x_.item_ptr(row, k_end);
    const half* w_ptr = w_.item_ptr(k, column);
    half* out_ptr = out_.item_ptr(row, column);

    if constexpr (use_half2 && !odd_rank)
    {
        half2* x_ptr2 = (half2*) x_ptr;
        half2* x_ptr2_end = (half2*) x_ptr_end;

        half2 r = {};

        while(x_ptr2 < x_ptr2_end)
        {
            half2 x_01 = *x_ptr2++;
            half2 x_23 = *x_ptr2++;
            half w_0 = *w_ptr; w_ptr += width;
            half w_1 = *w_ptr; w_ptr += width;
            half w_2 = *w_ptr; w_ptr += width;
            half w_3 = *w_ptr; w_ptr += width;
            half2 w_01 = __halves2half2(w_0, w_1);
            half2 w_23 = __halves2half2(w_2, w_3);
            r = __hfma2(x_01, w_01, r);
            r = __hfma2(x_23, w_23, r);
        }

        half rh = __hadd(r.x, r.y);

        __shared__ half accum[MAX_DIM_SMALL / S_BLOCKSIZE][S_THREADS_X];
        accum[threadIdx.y][threadIdx.x] = rh;
        __syncthreads();

        if (threadIdx.y == 0)
        {
            half acc = rh;
            for (int i = 1; i < blockDim.y; ++i) acc = __hadd(accum[i][threadIdx.x], acc);
            if (no_zero) acc = __hadd(acc, *out_ptr);
            *out_ptr = acc;
        }
    }
    else
    {
        half r = {};

        while(x_ptr < x_ptr_end)
        {
            if constexpr (odd_rank)
            {
                half x_item = *x_ptr++;
                half w_item = *w_ptr; w_ptr += width;
                r = __hfma(x_item, w_item, r);
            }
            else
            {
                #pragma unroll
                for (int i = 0; i < 4; ++i)
                {
                    half x_item = *x_ptr++;
                    half w_item = *w_ptr; w_ptr += width;
                    r = __hfma(x_item, w_item, r);
                }
            }
        }

        __shared__ half accum[MAX_DIM_SMALL / S_BLOCKSIZE][S_THREADS_X];
        accum[threadIdx.y][threadIdx.x] = r;
        __syncthreads();

        if (threadIdx.y == 0)
        {
            half acc = accum[0][threadIdx.x];
            for (int i = 1; i < blockDim.y; ++i) acc = __hadd(accum[i][threadIdx.x], acc);
            if (no_zero) acc = __hadd(acc, *out_ptr);
            *out_ptr = acc;
        }
    }
}

void half_matmul_small_cuda
(
    ExLlamaTuning* tuningParams,
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width,
    bool no_zero,
    cudaStream_t alt_stream
)
{
    bool use_half2 = !tuningParams->matmul_no_half2;

    //printf("kernel: (%i, %i) @ (%i, %i) -> (%i, %i)\n", height, dim, dim, width, height, width);

    dim3 threads
    (
        S_THREADS_X,
        (dim + S_BLOCKSIZE - 1) / S_BLOCKSIZE,
        1
    );

    dim3 blocks
    (
        (width + S_THREADS_X - 1) / S_THREADS_X,
        1,
        height
    );

    //printf("t... %i %i %i\n", threads.x, threads.y, threads.z);
    //printf("b... %i %i %i\n", blocks.x, blocks.y, blocks.z);
    //if (!no_zero) cudaMemsetAsync(out, 0, height * width * sizeof(half));

    if (dim & 0x03)
    {
        half_matmul_small_kernel<false, true> <<<blocks, threads, 0, alt_stream>>>(x, w, out, height, dim, width, no_zero);
    }
    else
    {
        if (use_half2) half_matmul_small_kernel<true,  false> <<<blocks, threads, 0, alt_stream>>>(x, w, out, height, dim, width, no_zero);
        else           half_matmul_small_kernel<false, false> <<<blocks, threads, 0, alt_stream>>>(x, w, out, height, dim, width, no_zero);
    }
}

