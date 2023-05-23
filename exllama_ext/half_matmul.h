#ifndef _half_matmul_h
#define _half_matmul_h

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

cudaError_t half_matmul_cublas_cuda
(
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width,
    cublasHandle_t handle
);

cudaError_t half_matmul_cuda
(
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width
);


#endif