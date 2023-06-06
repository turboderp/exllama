#ifndef _half_matmul_cuh
#define _half_matmul_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <ATen/cuda/CUDAContext.h>

// Workaround for hipify_python using rocblas instead of hipblas.
#if defined(USE_ROCM)
#include <hipblas/hipblas.h>
#define rocblas_handle hipblasHandle_t
#endif

void half_matmul_cuda
(
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width
);

void half_matmul_cublas_cuda
(
    const half* x,
    const half* w,
    half* out,
    const int height,
    const int dim,
    const int width,
    cublasHandle_t handle
);

#endif