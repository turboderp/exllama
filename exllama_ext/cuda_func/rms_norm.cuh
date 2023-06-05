#ifndef _rms_norm_cuh
#define _rms_norm_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "../tuning.h"

void rms_norm_cuda
(
    ExLlamaTuning* tuningParams,
    half* x,
    const half* w,
    half* out,
    const float epsilon,
    const int rows,
    const int dim,
    const int device_index
);

#endif