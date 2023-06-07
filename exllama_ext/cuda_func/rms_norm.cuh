#ifndef _rms_norm_cuh
#define _rms_norm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "../tuning.h"

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
);

void rms_norm_cuda_destroy_graph(const int device_index);

#endif