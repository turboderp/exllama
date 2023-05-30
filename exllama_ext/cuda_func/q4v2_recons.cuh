#ifndef _q4v2_recons_cuh
#define _q4v2_recons_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

cudaError_t q4v2_recons_cuda
(
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int width,
    const int groupsize,
    const uint16_t* seq_g_idx
);

#endif

