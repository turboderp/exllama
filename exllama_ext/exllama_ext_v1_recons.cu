// #include <torch/all.h>
// #include <torch/python.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT4 =  32;

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

//template <typename scalar_t>
__global__ void VecQuant4ReconsKernel_v1(
    const       int* __restrict__ mat,
           c10::Half* __restrict__ res,
    const  c10::Half* __restrict__ scales,
    const  c10::Half* __restrict__ zeros,
    int height,
    int width
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int n_rows = h * 8 + b;
  int n_cols = w;
  c10::Half scale = scales[w];
  c10::Half zero = zeros[w];
  int i = width * h + width * (b / 8) + w;
  int shift = b % 8 * 4;
  unsigned int tmp = as_unsigned(mat[i]);
  c10::Half result = (scale * c10::Half((tmp >> shift) & 0xF) - zero);
  res[n_rows * width + n_cols] = result;
}

void vecquant4recons_v1_cuda(
  torch::Tensor mat,
  torch::Tensor res,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = BLOCKWIDTH;
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(
//     scales.type(), "vecquant4recons_v1_cuda", ([&] {
      VecQuant4ReconsKernel_v1<<<blocks, threads>>>(
        mat.data_ptr<int>(),
        res.data_ptr<c10::Half>(),
        scales.data_ptr<c10::Half>(),
        zeros.data_ptr<c10::Half>(),
        height,
        width
      );
//     })
//   );
}
