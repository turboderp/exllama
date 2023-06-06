#ifndef _cuda_buffers_cuh
#define _cuda_buffers_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

const int CUDA_MAX_DEVICES = 16;

class CudaBuffers
{
public:
    int device;

    half* temp_state;           // [max_hidden_rows * intermediate_size]
    half* temp_mlp;             // [hidden_dim * intermediate_size]
    float* temp_zeros_float;    // [max_hidden_rows]
    half* temp_dq;              // size of largest quant tensor * 8

    int current_zeros_float;
    int max_zeros_float;

    CudaBuffers
    (
        int _device,
        half* _temp_state,
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
    half* _temp_mlp,
    float* _temp_zeros_float,
    half* _temp_dq,
    int _max_zeros_float
);

#endif