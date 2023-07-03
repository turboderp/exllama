#define _cuda_buffers_cu
#include "cuda_buffers.cuh"

CudaBuffers* g_buffers[CUDA_MAX_DEVICES] = {NULL};
// __constant__ half2 q4_table[16][256];
// half2 q4_table_host[16][256];
// bool q4_table_init = false;

CudaBuffers::CudaBuffers
(
    int _device,
    half* _temp_state,
    int _temp_state_size,
    half* _temp_mlp,
    float* _temp_zeros_float,
    half* _temp_dq,
    int _max_zeros_float
) :
    device(_device),
    temp_state(_temp_state),
    temp_state_size(_temp_state_size),
    temp_mlp(_temp_mlp),
    temp_zeros_float(_temp_zeros_float),
    temp_dq(_temp_dq),
    max_zeros_float(_max_zeros_float),
    current_zeros_float(0)
{
    cudaSetDevice(_device);

    cudaStreamCreate(&alt_stream_1);
    cudaStreamCreate(&alt_stream_2);
    cudaStreamCreate(&alt_stream_3);
    cudaEventCreate(&alt_stream_1_done);
    cudaEventCreate(&alt_stream_2_done);
    cudaEventCreate(&alt_stream_3_done);
}

CudaBuffers::~CudaBuffers()
{
    cudaStreamDestroy(alt_stream_1);
    cudaStreamDestroy(alt_stream_2);
    cudaStreamDestroy(alt_stream_3);
    cudaEventDestroy(alt_stream_1_done);
    cudaEventDestroy(alt_stream_2_done);
    cudaEventDestroy(alt_stream_3_done);
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
    int _temp_state_size,
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
        _temp_state_size,
        _temp_mlp,
        _temp_zeros_float,
        _temp_dq,
        _max_zeros_float
    );

    g_buffers[_device] = buffers;
}

void cleanup_buffers_cuda()
{
    for (int i = 0; i < CUDA_MAX_DEVICES; i++)
    {
        if (!g_buffers[i]) continue;
        delete g_buffers[i];
        g_buffers[i] = NULL;
    }
}
