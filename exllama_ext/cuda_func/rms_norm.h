#ifndef _rms_norm_h
#define _rms_norm_h

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

cudaError_t rms_norm_cuda
(
    half* x,
    const half* w,
    half* out,
    const float epsilon,
    const int rows,
    const int dim,
    const int device_index
);

#endif