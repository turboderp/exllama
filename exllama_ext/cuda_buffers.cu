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

//     if (!q4_table_init)
//     {
//         for (uint v_zero = 0; v_zero < 16; v_zero++)
//         {
//             for (uint v_read = 0; v_read < 256; v_read++)
//             {
//                 half v_0 = __float2half((float)((int)((v_read      ) & 0x0f) - v_zero - 1));
//                 half v_1 = __float2half((float)((int)((v_read >>  4) & 0x0f) - v_zero - 1));
//                 half2 v_01 = {v_0, v_1};
//                 q4_table_host[v_zero][v_read] = v_01;
//             }
//         }
//         q4_table_init = true;
//     }
//
//     cudaSetDevice(_device);
//     cudaMemcpyToSymbol(q4_table, q4_table_host, 16 * 256 * sizeof(half2));
//     cudaDeviceSynchronize();
}
