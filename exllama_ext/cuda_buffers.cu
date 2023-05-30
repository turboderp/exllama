#include "cuda_buffers.cuh"

CudaBuffers* g_buffers[CUDA_MAX_DEVICES] = {NULL};

CudaBuffers::CudaBuffers
(
    const int _rows,
    const int _mlp_rows,
    const int _intermediate_size,
    const int _hidden_size
) :
    rows(_rows),
    mlp_rows(_mlp_rows),
    intermediate_size(_intermediate_size),
    hidden_size(_hidden_size)
{
    cudaMalloc(&rms_norm_scratch, rows * sizeof(float));
    cudaMalloc(&mlp_temp, 2 * mlp_rows * intermediate_size * sizeof(half));
    cudaMalloc(&state_temp, rows * hidden_size * sizeof(half));
}

CudaBuffers::~CudaBuffers()
{
    cudaFree(rms_norm_scratch);
    cudaFree(mlp_temp);
    cudaFree(state_temp);
}

void CudaBuffers::zero_rms_norm_scratch(const int _rows)
{
    cudaMemsetAsync(rms_norm_scratch, 0, _rows * sizeof(float));
}

void CudaBuffers::zero_mlp_temp(const int _mlp_rows)
{
    cudaMemsetAsync(mlp_temp, 0, 2 * _mlp_rows * intermediate_size * sizeof(half));
}

void CudaBuffers::zero_state_temp(const int _rows)
{
    cudaMemsetAsync(state_temp, 0, rows * hidden_size * sizeof(half));
}

CudaBuffers* get_buffers(const int device_index)
{
    return g_buffers[device_index];
}

cudaError_t prepare_buffers_cuda
(
    const int device_index,
    const int rows,
    const int mlp_rows,
    const int intermediate_size,
    const int hidden_size
)
{
    cudaError_t _cuda_err = cudaSuccess;

    CudaBuffers* buffers = g_buffers[device_index];
    if (buffers) delete buffers;

    buffers = new CudaBuffers
    (
        rows,
        mlp_rows,
        intermediate_size,
        hidden_size
    );

    g_buffers[device_index] = buffers;

    return _cuda_err;
}

cudaError_t free_buffers_cuda
(
    const int device_index
)
{
    cudaError_t _cuda_err = cudaSuccess;

    CudaBuffers* buffers = g_buffers[device_index];
    if (buffers) delete buffers;
    g_buffers[device_index] = NULL;

    return _cuda_err;
}
