//#include <torch/all.h>
//#include <torch/python.h>
//#include <c10/cuda/CUDAGuard.h>
//#include <cuda_fp16.h//#include <torch/types.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// v1

//void vecquant4matmul_v1_cuda(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor scales, torch::Tensor zeros);
//void vecquant4matmul_v1(torch::Tensor vec, torch::Tensor mat, torch::Tensor mul, torch::Tensor scales, torch::Tensor zeros)
//{
//  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
//  vecquant4matmul_v1_cuda(vec, mat, mul, scales, zeros);
//}
//
//void vecquant4recons_v1_cuda(torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros);
//void vecquant4recons_v1(torch::Tensor mat, torch::Tensor res, torch::Tensor scales, torch::Tensor zeros)
//{
//  const at::cuda::OptionalCUDAGuard device_guard(device_of(scales));
//  vecquant4recons_v1_cuda(mat, res, scales, zeros);
//}

// v2

void q4v2_matmul_cuda
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
    const uint32_t* g_idx
);

void q4v2_matmul
(
    torch::Tensor x,
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor w_scales,
    torch::Tensor w_zeros,
    int groupsize,
    torch::Tensor g_idx
)
{
    TORCH_CHECK(x.dtype() == torch::kHalf, "x must be a half tensor");
    TORCH_CHECK(w.dtype() == torch::kInt, "w must be an int (q4) tensor");
    TORCH_CHECK(out.dtype() == torch::kHalf, "out must be a half tensor");
    TORCH_CHECK(w_scales.dtype() == torch::kHalf, "w_scales must be a half tensor");
    TORCH_CHECK(w_zeros.dtype() == torch::kInt, "w_zeros must be an int tensor");
    TORCH_CHECK(x.size(1) == w.size(0) * 8, "x and w have incompatible shapes");
    TORCH_CHECK(x.size(1) % 256 == 0, "x.shape[1] must be multiple of 256");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w_scales));

    int height = x.size(0);
    int dim = x.size(1);
    int width = w.size(1);

    q4v2_matmul_cuda
    (
        (half*) x.data_ptr(),
        (uint32_t*) w.data_ptr(),
        (half*) out.data_ptr(),
        (half*) w_scales.data_ptr(),
        (uint32_t*) w_zeros.data_ptr(),
        height,
        dim,
        width,
        groupsize,
        g_idx.device().is_meta() ? NULL : (uint32_t*) g_idx.data_ptr()
    );
}

void q4v2_recons_cuda
(
    const uint32_t* w,
    half* out,  // y
    const half* w_scales,
    const uint32_t* w_zeros,
    const int height,
    const int width,
    const int groupsize,
    const uint32_t* g_idx
);

void q4v2_recons
(
    torch::Tensor w,
    torch::Tensor out,
    torch::Tensor w_scales,
    torch::Tensor w_zeros,
    int groupsize,
    torch::Tensor g_idx
)
{
    TORCH_CHECK(w.dtype() == torch::kInt, "w must be an int (q4) tensor");
    TORCH_CHECK(out.dtype() == torch::kHalf, "out must be a half tensor");
    TORCH_CHECK(w_scales.dtype() == torch::kHalf, "w_scales must be a half tensor");
    TORCH_CHECK(w_zeros.dtype() == torch::kInt, "w_zeros must be an int tensor");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(w_scales));

    int height = w.size(0);
    int width = w.size(1);

    q4v2_recons_cuda
    (
        (uint32_t*) w.data_ptr(),
        (half*) out.data_ptr(),
        (half*) w_scales.data_ptr(),
        (uint32_t*) w_zeros.data_ptr(),
        height,
        width,
        groupsize,
        g_idx.device().is_meta() ? NULL : (uint32_t*) g_idx.data_ptr()
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
//  m.def("vecquant4matmul_v1", &vecquant4matmul_v1, "Vector 4-bit Quantized Matrix Multiplication (CUDA) v1");
//  m.def("vecquant4recons_v1", &vecquant4recons_v1, "Vector 4-bit Quantized Matrix Reconstruction (CUDA) v1");
  m.def("q4v2_matmul", &q4v2_matmul, "Vector 4-bit Quantized Matrix Multiplication (CUDA) v2");
  m.def("q4v2_recons", &q4v2_recons, "Vector 4-bit Quantized Matrix Reconstruction (CUDA) v2");
}