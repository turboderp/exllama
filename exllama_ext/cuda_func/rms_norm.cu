#include "rms_norm.cuh"
#include "../cuda_buffers.cuh"
#include "../util.cuh"
#include "../matrix.cuh"

const int THREADS_X = 16;
const int THREADS_Y = 4;
const int BLOCKSIZE_X = 4;

// scratch = sum(x * x, dim = -1)

__global__ void rms_norm_row_product_kernel
(
    half* x,
    float* scratch,
    const int rows,
    const int dim
)
{
    int column = (THREADS_X * blockIdx.x + threadIdx.x) * BLOCKSIZE_X;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= rows) return;
    if (column >= dim) return;

    // Accumulate

    float acc = 0.0f;
    int idx = row * dim + column;

    #pragma unroll
    for (int k = 0; k < BLOCKSIZE_X; k++)
    {
        float m = __half2float(x[idx++]);
        acc = fma(m, m, acc);
    }

    atomicAdd(&scratch[row], acc);
}

// x = x * w / sqrt(scratch / dim + epsilon)

__global__ void rms_norm_kernel
(
    half* x,
    const half* w,
    half* out,
    float* scratch,
    const float epsilon,
    const float r_dim,
    const int rows,
    const int dim
)
{
    int column = (THREADS_X * blockIdx.x + threadIdx.x) * BLOCKSIZE_X;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= rows) return;
    if (column >= dim) return;

    half2* x2_ptr = (half2*) &x[row * dim + column];
    half2* out2_ptr = (half2*) &out[row * dim + column];
    half2* w2_ptr = (half2*) &w[column];

    float rmf = rsqrtf(scratch[row] * r_dim + epsilon);
    half rm = __float2half_rn(rmf);
    half2 rm2 = __half2half2(rm);

    #pragma unroll
    for (int k = 0; k < BLOCKSIZE_X / 2; k++)
    {
        half2 m2 = *x2_ptr++;
        half2 w2 = *w2_ptr++;
        m2 = __hmul2(m2, rm2);
        m2 = __hmul2(m2, w2);
        *out2_ptr++ = m2;
    }
}

// x = x * w / sqrt(row_mean(x * x) + epsilon)
//
// works in-place if x == out

cudaError_t rms_norm_cuda
(
    half* x,
    const half* w,
    half* out,
    const float epsilon,
    const int rows,
    const int dim,
    const int device_index
)
{
    cudaError_t _cuda_err = cudaSuccess;

    CudaBuffers* buffers = get_buffers(device_index);
    buffers->zero_rms_norm_scratch(rows);

    float r_dim = 1.0f / (float) dim;

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (dim + THREADS_X - 1) / THREADS_X / BLOCKSIZE_X,
        (rows + THREADS_Y - 1) / THREADS_Y,
        1
    );

    rms_norm_row_product_kernel<<<blocks, threads>>>(x, buffers->rms_norm_scratch, rows, dim);
    rms_norm_kernel<<<blocks, threads>>>(x, w, out, buffers->rms_norm_scratch, epsilon, r_dim, rows, dim);

//_cuda_fail:

    //if (scratch) cudaFree(scratch);

    return _cuda_err;
}
