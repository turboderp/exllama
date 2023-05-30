#ifndef _q4v2_sequential_cuh
#define _q4v2_sequential_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>

cudaError_t q4v2_sequential_cuda
(
    uint32_t* w,
    const int w_height,
    const int w_width,
    const uint32_t* g_idx,  // size: w_height * 8
    uint16_t* seq_g_idx,    // size: w_height * 8 * 2
    uint32_t* x_map_cuda,   // size: w_height * 8
    const int num_groups
);

#endif