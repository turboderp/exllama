#ifndef _q4v2_matmul_cuh
#define _q4v2_matmul_cuh

#include "q4_matrix.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <ATen/cuda/CUDAContext.h>

cudaError_t q4v2_matmul_cuda
(
    const half* x,
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int dim,
    const int width,
    const int groupsize,
    const uint16_t* seq_g_idx,
    const uint32_t* x_map
);

void q4_matmul_cuda
(
    const half* x,
    const int x_height,
    const Q4Matrix* w,
    half* out
);

void q4_matmul_recons_cuda
(
    const half* x,
    const int x_height,
    Q4Matrix* w,
    half* out,
    const cublasHandle_t handle
);

#endif