//#include <torch/all.h>
//#include <torch/python.h>
//#include <c10/cuda/CUDAGuard.h>
//#include <cuda_fp16.h>
//#include <torch/types.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

// v1

void vecquant4matmul_v1_cuda(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor scales, torch::Tensor zeros);
void vecquant4matmul_v1(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor scales, torch::Tensor zeros)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_v1_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4recons_v1_cuda(torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros);
void vecquant4recons_v1(torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  vecquant4recons_v1_cuda(mat, res, scales, zeros);
}

// v2

void vecquant4matmul_v2_cuda(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor scales, torch::Tensor zeros, torch::Tensor g_idx, int vec_height);
void vecquant4matmul_v2(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor scales, torch::Tensor zeros, torch::Tensor g_idx, int vec_height)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_v2_cuda(vec, mat, mul, scales, zeros, g_idx, vec_height);
}

void vecquant4recons_v2_cuda(torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros, torch::Tensor g_idx);
void vecquant4recons_v2(torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros, torch::Tensor g_idx)
{
  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
  vecquant4recons_v2_cuda(mat, res, scales, zeros, g_idx);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("vecquant4matmul_v1", &vecquant4matmul_v1, "Vector 4-bit Quantized Matrix Multiplication (CUDA) v1");
  m.def("vecquant4matmul_v2", &vecquant4matmul_v2, "Vector 4-bit Quantized Matrix Multiplication (CUDA) v2");
  m.def("vecquant4recons_v1", &vecquant4recons_v1, "Vector 4-bit Quantized Matrix Reconstruction (CUDA) v1");
  m.def("vecquant4recons_v2", &vecquant4recons_v2, "Vector 4-bit Quantized Matrix Reconstruction (CUDA) v2");
}