#ifndef _q4v2_mlp_h
#define _q4v2_mlp_h

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

cudaError_t q4v2_mlp_cuda
(
    const half* x,
    half* out,
    
    const uint32_t* gate,
    const half* gate_scales,
    const uint32_t* gate_zeros,
    const uint16_t* gate_seq_g_idx,
    const uint32_t* gate_x_map,

    const uint32_t* up,
    const half* up_scales,
    const uint32_t* up_zeros,
    const uint16_t* up_seq_g_idx,
    const uint32_t* up_x_map,

    const int height,
    const int dim,
    const int width,
    const int groupsize
);

#endif