// #include <torch/all.h>
// #include <torch/python.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT4 = 32;

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

#define AT_DISPATCH_HALF_AND_INT_TYPES(TYPE, NAME, ...)                           \
  [&] {                                                                           \
    const at::ScalarType _st = TYPE;                                              \
    switch (_st) {                                                                \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Half, c10::Half, __VA_ARGS__)          \
      AT_PRIVATE_CASE_TYPE(at::ScalarType::Int, int, __VA_ARGS__)                 \
      default:                                                                    \
        AT_ERROR("Unsupported dispatch to ", #NAME, " for given scalar type");    \
    }                                                                             \
  }()

// template <typename scalar_t>
__global__ void VecQuant4ReconsKernel_v2(
    const       int* __restrict__ mat,
           c10::Half* __restrict__ res,
    const  c10::Half* __restrict__ scales,
    const       int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
    int height,
    int width,
    int zero_width
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int n_rows = h * 8 + b;
  int n_cols = w;
  int z_rows = as_int(g_idx[n_rows]);
  int z_cols = n_cols / 8;
  int z_shift = (n_cols % 8) * 4;
  half scale = scales[z_rows * width + n_cols];

  float scale_f = __half2float(scale);
  c10::Half zero = scale_f * c10::Half(((as_unsigned(zeros[z_rows * zero_width + z_cols]) >> z_shift) & 0xF) + 1);
  int i = width * h + width * (b / 8) + w;
  int shift = b % 8 * 4;
  unsigned int tmp = as_unsigned(mat[i]);
  c10::Half result = (scale_f * c10::Half((tmp >> shift) & 0xF) - zero);
  res[n_rows * width + n_cols] = result;
}

void vecquant4recons_v2_cuda(
  torch::Tensor mat,
  torch::Tensor res,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = BLOCKWIDTH;
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

    VecQuant4ReconsKernel_v2<<<blocks, threads>>>(
        mat.data_ptr<int>(),
        res.data_ptr<c10::Half>(),
        scales.data_ptr<c10::Half>(),
        zeros.data_ptr<int>(),
        g_idx.data_ptr<int>(),
        height,
        width,
        zero_width
      );

//
//
//
//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//     scales.type(), "vecquant4recons_v2_cuda", ([&] {
//       VecQuant4ReconsKernel_v2<<<blocks, threads>>>(
//         mat.data_ptr<int>(),
//         res.data_ptr<c10::Half>(),
//         scales.data_ptr<c10::Half>(),
//         zeros.data_ptr<int>(),
//         g_idx.data_ptr<int>(),
//         height,
//         width,
//         zero_width
//       );
//     })
//  );
}