#ifndef _q4v2_mlp_cuh
#define _q4v2_mlp_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

cudaError_t q4v2_mlp_cuda
(
    half* x,                        // shape == (height, dim)
    half* out,                      // shape == x.shape

    const half* rms_norm_weight,    // shape == (x.shape[1],) == (dim,)
    float epsilon,

    const uint32_t* gate,           // shape == (dim, width)
    const half* gate_scales,
    const uint32_t* gate_zeros,
    const uint16_t* gate_seq_g_idx,
    const uint32_t* gate_x_map,
    const int gate_groupsize,

    const uint32_t* up,
    const half* up_scales,
    const uint32_t* up_zeros,
    const uint16_t* up_seq_g_idx,
    const uint32_t* up_x_map,
    const int up_groupsize,

    const uint32_t* down,
    const half* down_scales,
    const uint32_t* down_zeros,
    const uint16_t* down_seq_g_idx,
    const uint32_t* down_x_map,
    const int down_groupsize,

    const int height,
    const int dim,
    const int width,

    const int device_index
);

#endif