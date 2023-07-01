#include "rms_norm.cuh"
#include "../cuda_buffers.cuh"
#include "../util.cuh"
#include "../matrix.cuh"

const int THREADS_X = 32;
const int THREADS_Y = 8;
const int BLOCKSIZE_X = 16;

// scratch = sum(x * x, dim = -1)

typedef void (*fp_rms_norm_row_product_kernel)
(
    half*,
    float*,
    const int,
    const int
);

template<bool use_half2>
__global__ void rms_norm_row_product_kernel
(
    half* __restrict__ x,
    float* __restrict__ scratch,
    const int rows,
    const int dim
)
{
    int column = (THREADS_X * blockIdx.x + threadIdx.x) * BLOCKSIZE_X;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= rows) return;
    if (column >= dim) return;

//     if (column == 0)
//     {
//         scratch[row] = 0.0f;
//         __syncthreads();
//     }

    float acc = 0.0f;
    int idx = row * dim + column;

    // Accumulate

    if constexpr (use_half2)
    {
        half2* x_ptr = (half2*) &x[idx];

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE_X / 2; k++)
        {
            half2 x2 = *x_ptr++;
            float m0 = __half2float(x2.x);
            float m1 = __half2float(x2.y);
            acc = fma(m0, m0, acc);
            acc = fma(m1, m1, acc);
        }
    }
    else
    {
        half* x_ptr = x + idx;

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE_X; k++)
        {
            float m0 = __half2float(*x_ptr++);
            acc = fma(m0, m0, acc);
        }
    }

//     // Use Warp Shuffle to accumulate within the warp
//
//     for (int offset = warpSize / 2; offset > 0; offset /= 2)
//         acc += __shfl_down_sync(0xffffffff, acc, offset);
//     if (threadIdx.x % warpSize == 0)
//         atomicAdd(&scratch[row], acc);

    atomicAdd(&scratch[row], acc);
}

// x = x * w / sqrt(scratch / dim + epsilon)

typedef void (*fp_rms_norm_kernel)
(
    half*,
    const half*,
    half*,
    float*,
    const float,
    const float,
    const int,
    const int
);

template<bool use_half2>
__global__ void rms_norm_kernel
(
    half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ out,
    float* __restrict__ scratch,
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

    float rmf = rsqrtf(scratch[row] * r_dim + epsilon);
    half rm = __float2half_rn(rmf);
    half2 rm2 = __half2half2(rm);

    if constexpr (use_half2)
    {
        half2* x2_ptr = (half2*) &x[row * dim + column];
        half2* out2_ptr = (half2*) &out[row * dim + column];
        const half2* w2_ptr = (const half2*) &w[column];

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
    else
    {
        half* x_ptr = &x[row * dim + column];
        half* out_ptr = &out[row * dim + column];
        const half* w_ptr = &w[column];

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE_X; k++)
        {
            half m = *x_ptr++;
            half w = *w_ptr++;
            m = __hmul(m, rm);
            m = __hmul(m, w);
            *out_ptr++ = m;
        }
    }

//     __syncthreads();
//     if (column >= dim - BLOCKSIZE_X) scratch[row] = 0.0f;
}

fp_rms_norm_row_product_kernel rms_norm_row_product_kernel_pick(ExLlamaTuning* tuningParams)
{
    // <bool use_half2>
    if (tuningParams->matmul_no_half2) {
        return rms_norm_row_product_kernel<false>;
    } else {
        return rms_norm_row_product_kernel<true>;
    }
};

fp_rms_norm_kernel rms_norm_kernel_pick(ExLlamaTuning* tuningParams)
{
    // <bool use_half2>
    if (tuningParams->matmul_no_half2) {
        return rms_norm_kernel<false>;
    } else {
        return rms_norm_kernel<true>;
    }
};

// x = x * w / sqrt(row_mean(x * x) + epsilon)
//
// works in-place if x == out

void rms_norm_cuda
(
    ExLlamaTuning* tuningParams,
    half* x,
    const half* w,
    half* out,
    const float epsilon,
    const int rows,
    const int dim,
    const int device_index
)
{
    CudaBuffers* buffers = get_buffers(device_index);
    float* temp = buffers->get_zeros_float(rows);

    float r_dim = 1.0f / (float) dim;

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        ((dim + THREADS_X - 1) / THREADS_X + THREADS_X - 1) / BLOCKSIZE_X,
        (rows + THREADS_Y - 1) / THREADS_Y,
        1
    );

    //cudaMemsetAsync(temp, 0, rows * sizeof(float));

    fp_rms_norm_row_product_kernel kernel1 = rms_norm_row_product_kernel_pick(tuningParams);
    kernel1<<<blocks, threads>>>(x, temp, rows, dim);

    fp_rms_norm_kernel kernel2 = rms_norm_kernel_pick(tuningParams);
    kernel2<<<blocks, threads>>>(x, w, out, temp, epsilon, r_dim, rows, dim);

    //cudaMemsetAsync(temp, 0, rows * sizeof(float));
}
