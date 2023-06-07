#include "rms_norm.cuh"
#include "../cuda_buffers.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#if defined(USE_ROCM)
#include "../hip_compat.cuh"
#endif

const int THREADS_X = 32;
const int THREADS_Y = 8;
const int BLOCKSIZE_X = 16;

// scratch = sum(x * x, dim = -1)

template<bool use_half2>
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

template<bool use_half2>
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
}

// x = x * w / sqrt(row_mean(x * x) + epsilon)
//
// works in-place if x == out

struct RMSNormDeviceContext
{
    bool rms_norm_graph_init = false;
    cudaGraph_t rms_norm_graph;
    cudaGraphNode_t rms_norm_node0;
    cudaGraphNode_t rms_norm_node1;
    cudaGraphNode_t rms_norm_node2;
    cudaGraphExec_t rms_norm_graphExec;
};

RMSNormDeviceContext contexts[CUDA_MAX_DEVICES] = {0};

void (*rms_norm_kernel_func1)(half*, float*, const int, const int);
void (*rms_norm_kernel_func2)(half*, const half*, half*, float*, const float, const float, const int, const int);

void rms_norm_cuda
(
    ExLlamaTuning* tuningParams,
    cudaStream_t stream,
    half* x,
    half* w,
    half* out,
    float epsilon,
    int rows,
    int dim,
    const int device_index
)
{
    cudaSetDevice(device_index);

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (dim + THREADS_X - 1) / THREADS_X / BLOCKSIZE_X,
        (rows + THREADS_Y - 1) / THREADS_Y,
        1
    );

    CudaBuffers* buffers = get_buffers(device_index);
    float* temp = buffers->temp_rms_norm;

    // cudaMemsetAsync(temp, 0, rows * sizeof(float));

    cudaMemsetParams memsetParams = {0};
    bool updateMemset = false;

    if (rows != buffers->last_rms_norm_rows[device_index] || !contexts[device_index].rms_norm_graph_init)
    {
        memsetParams.dst = (void*)temp;
        memsetParams.value = 0;
        memsetParams.pitch = 0;
        memsetParams.elementSize = sizeof(float);
        memsetParams.width = rows;
        memsetParams.height = 1;

        updateMemset = true;
        buffers->last_rms_norm_rows[device_index] = rows;
    }

    if (tuningParams->rmsnorm_no_half2)
    {
         rms_norm_kernel_func1 = &rms_norm_row_product_kernel<false>;
         rms_norm_kernel_func2 = &rms_norm_kernel<false>;
    }
    else
    {
         rms_norm_kernel_func1 = &rms_norm_row_product_kernel<true>;
         rms_norm_kernel_func2 = &rms_norm_kernel<true>;
    }

    float r_dim = 1.0f / (float) dim;

    void* args1[] = { &x, &temp, &rows, &dim };
    void* args2[] = { &x, &w, &out, &temp, &epsilon, &r_dim, &rows, &dim };

    cudaKernelNodeParams params1 = { .func           = (void *)rms_norm_kernel_func1,
                                     .gridDim        = blocks,
                                     .blockDim       = threads,
                                     .sharedMemBytes = 0,
                                     .kernelParams   = args1,
                                     .extra          = nullptr };
    cudaKernelNodeParams params2 = { .func           = (void *)rms_norm_kernel_func2,
                                     .gridDim        = blocks,
                                     .blockDim       = threads,
                                     .sharedMemBytes = 0,
                                     .kernelParams   = args2,
                                     .extra          = nullptr };

    if (!contexts[device_index].rms_norm_graph_init)
    {
        cudaGraphCreate(&contexts[device_index].rms_norm_graph, 0);

        cudaGraphAddMemsetNode(&contexts[device_index].rms_norm_node0, contexts[device_index].rms_norm_graph, nullptr, 0, &memsetParams);
        cudaGraphAddKernelNode(&contexts[device_index].rms_norm_node1, contexts[device_index].rms_norm_graph, &contexts[device_index].rms_norm_node0, 1, &params1);
        cudaGraphAddKernelNode(&contexts[device_index].rms_norm_node2, contexts[device_index].rms_norm_graph, &contexts[device_index].rms_norm_node1, 1, &params2);

        cudaGraphInstantiate(&contexts[device_index].rms_norm_graphExec, contexts[device_index].rms_norm_graph, nullptr, nullptr, 0);

        contexts[device_index].rms_norm_graph_init = true;
    }
    else
    {
        if (updateMemset)
            cudaGraphExecMemsetNodeSetParams(contexts[device_index].rms_norm_graphExec, contexts[device_index].rms_norm_node0, &memsetParams);

        cudaGraphExecKernelNodeSetParams(contexts[device_index].rms_norm_graphExec, contexts[device_index].rms_norm_node1, &params1);
        cudaGraphExecKernelNodeSetParams(contexts[device_index].rms_norm_graphExec, contexts[device_index].rms_norm_node2, &params2);
    }

    cudaGraphLaunch(contexts[device_index].rms_norm_graphExec, stream);
}

void rms_norm_cuda_destroy_graph(const int device_index)
{
    // TODO: Find a way to actually call this. The EXLlama destructor isn't called until after the extension is
    // unloaded. Or just let the CUDA runtime clean up.

    cudaSetDevice(device_index);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(contexts[device_index].rms_norm_graphExec);
    cudaGraphDestroy(contexts[device_index].rms_norm_graph);
}