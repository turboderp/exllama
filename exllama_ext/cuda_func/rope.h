#ifndef _rope_h
#define _rope_h

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

cudaError_t rope_cuda
(
    half* x,
    const half* sin,
    const half* cos,
    const int rows,
    const int head_dim,
    const int num_heads,
    const int past_len
);

#endif