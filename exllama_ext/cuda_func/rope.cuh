#ifndef _rope_cuh
#define _rope_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "../tuning.h"

void rope_cuda
(
    ExLlamaTuning* tuningParams,
    half* x,
    const half* sin,
    const half* cos,
    const int bsz,
    const int rows,
    const int head_dim,
    const int num_heads,
    const int past_len,
    cudaStream_t alt_stream = NULL
);

#endif