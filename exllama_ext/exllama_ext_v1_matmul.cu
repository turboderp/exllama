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

// template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_v1(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           c10::Half* __restrict__ mul,
    const  c10::Half* __restrict__ scales,
    const  c10::Half* __restrict__ zeros,
	  int batch,
	  int vec_height,
    int height,
    int width
) {
  const int blockwidth2 = BLOCKWIDTH / 2;
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ half2 blockvec[blockwidth2];
  if (threadIdx.x < blockwidth2)
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

  __shared__ half2 deq2[256][8];
  int val = threadIdx.x / 8;
  int off = threadIdx.x % 8;
  for (; val < 256; val += BLOCKWIDTH / 8) {
    deq2[val][off] = __halves2half2(
       __int2half_rn(val & 0xF), __int2half_rn(val >> 4)
    );
  }

  int i = width * h + w;
  int k = 0;

  c10::Half res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
	  c10::Half scale_f = scales[w];
    c10::Half zero_f = zeros[w];
    half2 scale = __half2half2(scale_f);
    half2 zero = __half2half2(-zero_f);

    res2 = {};
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scale, zero), blockvec[k + 3], res2);
	  i += width;
    k += 4;
    res = __hadd(res, __hadd(res2.x, res2.y));
  }

  __half* mul2 = (__half*)mul;
  atomicAdd(&mul2[b * width + w], res);
}

void vecquant4matmul_v1_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1) / 2;
  int height = mat.size(0);
  int width = mat.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

//   AT_DISPATCH_SWITCH(vec.type(), "vecquant4matmul_v1_cuda",
//     AT_DISPATCH_CASE(at::ScalarType::Half, ([&] {
      VecQuant4MatMulKernel_v1<<<blocks, threads>>>(
        (half2*) vec.data_ptr<c10::Half>(),
        mat.data_ptr<int>(),
        mul.data_ptr<c10::Half>(),
        scales.data_ptr<c10::Half>(),
        zeros.data_ptr<c10::Half>(),
        batch, vec_height, height, width
      );
//     })
//   ));
}