#ifndef _q4_mlp_cuh
#define _q4_mlp_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "q4_matrix.cuh"

void q4_mlp_cuda
(
    half* x,                        // shape == (height, dim)
    half* out,                      // shape == (height, dim)
    const half* rms_norm_weight,    // shape == (x.shape[1],) == (dim,)
    float epsilon,
    Q4Matrix* gate,
    Q4Matrix* up,
    Q4Matrix* down,
    const int height,
    const int dim,
    const int device_index
);

#endif