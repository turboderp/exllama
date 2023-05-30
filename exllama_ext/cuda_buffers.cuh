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
    int rows;
    int mlp_rows;
    int intermediate_size;
    int hidden_size;

    float* rms_norm_scratch;    // [rows]
    half* mlp_temp;             // [2 * mlp_rows * intermediate_size]
    half* state_temp;           // [rows * hidden_size]

    CudaBuffers
    (
        const int _rows,
        const int _mlp_rows,
        const int _intermediate_size,
        const int _hidden_size
    );
    ~CudaBuffers();

    void zero_rms_norm_scratch(const int _rows);
    void zero_mlp_temp(const int _mlp_rows);
    void zero_state_temp(const int _rows);
};

CudaBuffers* get_buffers(const int device_index);

cudaError_t prepare_buffers_cuda
(
    const int device_index,
    const int rows,
    const int mlp_rows,
    const int intermediate_size,
    const int hidden_size
);

cudaError_t free_buffers_cuda
(
    const int device_index
);

#endif