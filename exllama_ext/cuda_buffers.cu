#include "cuda_buffers.cuh"

CudaBuffers* g_buffers[CUDA_MAX_DEVICES] = {NULL};

CudaBuffers::CudaBuffers
(
    int _device,
    half* _temp_state,
    half* _temp_mlp,
    float* _temp_zeros_float,
    half* _temp_dq,
    int _max_zeros_float
) :
    device(_device),
    temp_state(_temp_state),
    temp_mlp(_temp_mlp),
    temp_zeros_float(_temp_zeros_float),
    temp_dq(_temp_dq),
    max_zeros_float(_max_zeros_float),
    current_zeros_float(0)
{
}

CudaBuffers::~CudaBuffers()
{
}

float* CudaBuffers::get_zeros_float(const int num_zeros)
{
    if (current_zeros_float + num_zeros >= max_zeros_float)
    {
        current_zeros_float = 0;
        cudaMemsetAsync(temp_zeros_float, 0, max_zeros_float * sizeof(float));
    }

    float* zeros = temp_zeros_float + current_zeros_float;
    current_zeros_float += num_zeros;
    return zeros;
}

CudaBuffers* get_buffers(const int device_index)
{
    return g_buffers[device_index];
}

void prepare_buffers_cuda
(
    int _device,
    half* _temp_state,
    half* _temp_mlp,
    float* _temp_zeros_float,
    half* _temp_dq,
    int _max_zeros_float
)
{
    CudaBuffers* buffers = new CudaBuffers
    (
        _device,
        _temp_state,
        _temp_mlp,
        _temp_zeros_float,
        _temp_dq,
        _max_zeros_float
    );

    g_buffers[_device] = buffers;
}
