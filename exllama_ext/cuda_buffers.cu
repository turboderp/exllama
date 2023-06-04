#include "cuda_buffers.cuh"

CudaBuffers* g_buffers[CUDA_MAX_DEVICES] = {NULL};

CudaBuffers::CudaBuffers
(
    int _device,
    half* _temp_state,
    half* _temp_mlp,
    float* _temp_rms_norm,
    half* _temp_dq
) :
    device(_device),
    temp_state(_temp_state),
    temp_mlp(_temp_mlp),
    temp_rms_norm(_temp_rms_norm),
    temp_dq(_temp_dq)
{
}

CudaBuffers::~CudaBuffers()
{
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
    float* _temp_rms_norm,
    half* _temp_dq
)
{
    CudaBuffers* buffers = new CudaBuffers
    (
        _device,
        _temp_state,
        _temp_mlp,
        _temp_rms_norm,
        _temp_dq
    );

    g_buffers[_device] = buffers;
}
