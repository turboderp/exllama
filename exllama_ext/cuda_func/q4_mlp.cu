#include "q4_mlp.cuh"
#include "q4_matmul.cuh"
#include "half_matmul.cuh"
#include "rms_norm.cuh"
#include "../cuda_buffers.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#if defined(USE_ROCM)
#include "../hip_compat.cuh"
#endif

const int THREADS_X = 32;
const int THREADS_Y = 4;
// const int MAX_DIMENSION = 8192;

__device__ __forceinline__ half silu(half x)
{
    half one = __float2half(1.0f);
    half neg_x = __hneg(x);
    half e = hexp(neg_x);
    half sum = __hadd(one, e);
    half r = hrcp(sum);
    half result = __hmul(x, r);
    return result;
}

__device__ __forceinline__ half2 silu(half2 x)
{
    half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 e = h2exp(neg_x);
    half2 sum = __hadd2(one, e);
    half2 r = h2rcp(sum);
    half2 result = __hmul2(x, r);
    return result;
}

typedef void (*fp_silu_mul_cuda_kernel)
(
    half*,
    const half*,
    const int,
    const int
);

template <bool use_half2>
__global__ void silu_mul_cuda_kernel
(
    half* __restrict__ x,
    const half* __restrict__ y,
    const int height,
    const int width
)
{
    MatrixView_half_rw x_(x, height, width);
    MatrixView_half y_(y, height, width);

    int column = (THREADS_X * blockIdx.x + threadIdx.x); if constexpr (use_half2) column *= 2;
    int row = THREADS_Y * blockIdx.y + threadIdx.y;
    if (row >= height) return;

    // silu(x) * y

    if constexpr (use_half2)
    {
        half2 one = __half2half2(__float2half(1.0f));

        half2 x_item = x_.item_half2(row, column);
        half2 y_item = y_.item_half2(row, column);

        x_item = silu(x_item);
        x_item = __hmul2(x_item, y_item);

        x_.set_half2(row, column, x_item);
    }
    else
    {
        half one = __float2half(1.0f);

        half x_item = x_.item(row, column);
        half y_item = y_.item(row, column);

        x_item = silu(x_item);
        x_item = __hmul(x_item, y_item);

        x_.set(row, column, x_item);
    }
}

fp_silu_mul_cuda_kernel silu_mul_cuda_kernel_pick(ExLlamaTuning* tuningParams)
{
    // <bool use_half2>
    if (tuningParams->matmul_no_half2) {
        return silu_mul_cuda_kernel<false>;
    } else {
        return silu_mul_cuda_kernel<true>;
    }
};

void q4_mlp_cuda
(
    ExLlamaTuning* tuningParams,
    half* x,                        // shape == (height, dim)
    const half* rms_norm_weight,    // shape == (x.shape[1],) == (dim,)
    float epsilon,
    Q4Matrix* gate,
    Q4Matrix* up,
    Q4Matrix* down,
    const int height,
    const int dim,
    const half* gate_a,
    const half* gate_b,
    const int gate_rank,
    const half* up_a,
    const half* up_b,
    const int up_rank,
    const half* down_a,
    const half* down_b,
    const int down_rank,
    half* lora_temp,
    cublasHandle_t handle,
    const int device_index
)
{
    CudaBuffers* buffers = get_buffers(device_index);

    // temp_x = rms_layernorm(x)

    half* temp_x = buffers->temp_state + height * dim;  // TOOD: ..
    TORCH_CHECK(buffers->temp_state_size >= 2 * height * dim, "temp_state buffer too small");
    rms_norm_cuda(tuningParams, x, rms_norm_weight, temp_x, epsilon, height, dim, device_index);

    // temp_mlp[0] = temp_x @ gate
    // temp_mlp[1] = temp_x @ up

    half* temp_mlp_0 = buffers->temp_mlp;
    half* temp_mlp_1 = buffers->temp_mlp + height * up->width;
    int temp_mlp_width = up->width;

    if (gate_a)
    {
        half_matmul_cublas_cuda(tuningParams, temp_x, gate_a, lora_temp, height, dim, gate_rank, handle);
        half_matmul_cublas_cuda(tuningParams, lora_temp, gate_b, temp_mlp_0, height, gate_rank, temp_mlp_width, handle);
    }
    if (up_a)
    {
        half_matmul_cublas_cuda(tuningParams, temp_x, up_a, lora_temp, height, dim, up_rank, handle);
        half_matmul_cublas_cuda(tuningParams, lora_temp, up_b, temp_mlp_1, height, up_rank, temp_mlp_width, handle);
    }

    if (!tuningParams->concurrent_streams)
    {
        q4_matmul_cuda(tuningParams, temp_x, height, gate, temp_mlp_0, gate_a ? true : false);
        q4_matmul_cuda(tuningParams, temp_x, height, up, temp_mlp_1, up_a ? true : false);
    }
    else
    {
        cudaStream_t str_1 = buffers->alt_stream_1;
        cudaStream_t str_2 = buffers->alt_stream_2;
        cudaEvent_t sync_1 = buffers->alt_stream_1_done;
        cudaEvent_t sync_2 = buffers->alt_stream_2_done;

        q4_matmul_cuda(tuningParams, temp_x, height, gate, buffers->temp_mlp, gate_a ? true : false, str_1);
        cudaEventRecord(sync_1, str_1);

        q4_matmul_cuda(tuningParams, temp_x, height, up, buffers->temp_mlp + height * up->width, up_a ? true : false, str_2);
        cudaEventRecord(sync_2, str_2);

        cudaStreamWaitEvent(NULL, sync_1, 0);
        cudaStreamWaitEvent(NULL, sync_2, 0);
    }

    // temp_mlp[0] = silu(temp_mlp[0]) * temp_mlp[1]

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (up->width + THREADS_X - 1) / THREADS_X / (tuningParams->silu_no_half2 ? 1 : 2),
        (height + THREADS_Y - 1) / THREADS_Y,
        1
    );

    fp_silu_mul_cuda_kernel kernel = silu_mul_cuda_kernel_pick(tuningParams);
    kernel<<<blocks, threads>>>(temp_mlp_0, temp_mlp_1, height, temp_mlp_width);

    // x += temp1 @ down (implicitly add the residual connection by not zeroing the output in the matmul)

    if (down_a)
    {
        half_matmul_cublas_cuda(tuningParams, temp_mlp_0, down_a, lora_temp, height, temp_mlp_width, down_rank, handle);
        half_matmul_cublas_cuda(tuningParams, lora_temp, down_b, x, height, down_rank, dim, handle, true);
    }
    q4_matmul_cuda(tuningParams, temp_mlp_0, height, down, x, true);

    // Reset the temp buffer after use so it's always zeros.
    //cudaMemsetAsync(buffers->temp_mlp, 0, 2 * height * up->width * sizeof(half));

}