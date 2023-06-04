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

    half* temp_state;           // [max_hidden_rows * hidden_dim]
    half* temp_mlp;             // [hidden_dim * intermediate_size]
    float* temp_rms_norm;       // [max_hidden_rows]

    CudaBuffers
    (
        int _device,
        half* _temp_state,
        half* _temp_mlp,
        float* _temp_rms_norm
    );
    ~CudaBuffers();
};

CudaBuffers* get_buffers(const int device_index);

void prepare_buffers_cuda
(
    int _device,
    half* _temp_state,
    half* _temp_mlp,
    float* _temp_rms_norm
);

#endif