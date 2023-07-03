#ifndef _cuda_buffers_cuh
#define _cuda_buffers_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

const int CUDA_MAX_DEVICES = 16;

// #ifndef _cuda_buffers_cu
// extern __constant__ half2 q4_table[16][256];
// #endif

class CudaBuffers
{
public:
    int device;

    half* temp_state;           // [max_hidden_rows * intermediate_size]
    int temp_state_size;
    half* temp_mlp;             // [hidden_dim * intermediate_size]
    float* temp_zeros_float;    // [max_hidden_rows]
    half* temp_dq;              // size of largest quant tensor * 8

    int current_zeros_float;
    int max_zeros_float;

    cudaStream_t alt_stream_1;
    cudaStream_t alt_stream_2;
    cudaStream_t alt_stream_3;
    cudaEvent_t alt_stream_1_done;
    cudaEvent_t alt_stream_2_done;
    cudaEvent_t alt_stream_3_done;

    CudaBuffers
    (
        int _device,
        half* _temp_state,
        int _temp_state_size,
        half* _temp_mlp,
        float* _temp_zeros_float,
        half* _temp_dq,
        int _max_zeros_float
    );
    ~CudaBuffers();

    float* get_zeros_float(const int num_zeros);
};

CudaBuffers* get_buffers(const int device_index);

void prepare_buffers_cuda
(
    int _device,
    half* _temp_state,
    int _temp_state_size,
    half* _temp_mlp,
    float* _temp_zeros_float,
    half* _temp_dq,
    int _max_zeros_float
);

void cleanup_buffers_cuda();

#endif