#ifndef _q4v2_recons_h
#define _q4v2_recons_h

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

void q4v2_recons_cuda
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

