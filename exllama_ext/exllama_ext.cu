#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT4 =  32;

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

// template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_v2(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           c10::Half* __restrict__ mul,
    const  c10::Half* __restrict__ scales,
    const  	 int* __restrict__ zeros,
    const  	 int* __restrict__ g_idx,
	  int batch,
	  int vec_height,
    int height,
    int width,
    int zero_width
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
  int g_h = h * 8;
  int k = 0;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  half res = __float2half(0.0f);
  half2 res2;

  unsigned int tmp;

  half* scales2 = (half*)scales;

  __syncthreads();

  while (k < blockwidth2) {
    int g = as_int(g_idx[g_h + (k * 2)]);
	half scale_h = scales2[g * width + w];
    half2 scale = __half2half2(scale_h);
    half2 zero = __half2half2(__hmul(__hmul(scale_h , __float2half(-1.0f)), __int2half_rn(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xF) + 1)));

    res2 = {};
    tmp = as_unsigned(mat[i]);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  0) & 0xff][off], scale, zero), blockvec[k + 0], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >>  8) & 0xff][off], scale, zero), blockvec[k + 1], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 16) & 0xff][off], scale, zero), blockvec[k + 2], res2);
    res2 = __hfma2(__hfma2(deq2[(tmp >> 24) & 0xff][off], scale, zero), blockvec[k + 3], res2);
	  i += width;
    k += 4;
    res = __hadd(res, __hadd(res2.x, res2.y));;
  }

  half* mul2 = (half*)mul;
  atomicAdd(&mul2[b * width + w], res);
}

void vecquant4matmul_v2_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx,
  int vec_height
) {
  int batch = vec.size(0);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    batch
  );
  dim3 threads(BLOCKWIDTH);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vec.type(), "vecquant4matmul_faster_cuda",
      ([&] {
          VecQuant4MatMulKernel_v2<<<blocks, threads>>>(
            (half2*) vec.data_ptr(),
            mat.data_ptr<int>(),
            mul.data_ptr<c10::Half>(),
            scales.data_ptr<c10::Half>(),
            zeros.data_ptr<int>(),
            g_idx.data_ptr<int>(),
            batch, vec_height, height, width, zero_width
          );
         })
     );
}

template <typename scalar_t>
__global__ void VecQuant4ReconsKernel_v2(
    const       int* __restrict__ mat,
           scalar_t* __restrict__ res,
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
  scalar_t zero = scale_f * scalar_t(((as_unsigned(zeros[z_rows * zero_width + z_cols]) >> z_shift) & 0xF) + 1);
  int i = width * h + width * (b / 8) + w;
  int shift = b % 8 * 4;
  unsigned int tmp = as_unsigned(mat[i]);
  scalar_t result = (scale_f * scalar_t((tmp >> shift) & 0xF) - zero);
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    scales.type(), "vecquant4recons_v2_cuda", ([&] {
      VecQuant4ReconsKernel_v2<<<blocks, threads>>>(
        mat.data<int>(),
        res.data<scalar_t>(),
        scales.data<c10::Half>(),
        zeros.data<int>(),
        g_idx.data<int>(),
        height,
        width,
        zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant4MatMulKernel_v1(
    const  half2* __restrict__ vec,
    const    int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
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

  scalar_t res = 0;
  half2 res2;

  unsigned int tmp;

  __syncthreads();

  while (k < blockwidth2) {
	  scalar_t scale_f = scales[w];
    scalar_t zero_f = zeros[w];
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

  AT_DISPATCH_SWITCH(vec.type(), "vecquant4matmul_v1_cuda",
    AT_DISPATCH_CASE(at::ScalarType::Half, ([&] {
      VecQuant4MatMulKernel_v1<<<blocks, threads>>>(
        (half2*) vec.data_ptr<scalar_t>(),
        mat.data_ptr<int>(),
        mul.data_ptr<scalar_t>(),
        scales.data_ptr<scalar_t>(),
        zeros.data_ptr<scalar_t>(),
        batch, vec_height, height, width
      );
    })
  ));
}

template <typename scalar_t>
__global__ void VecQuant4ReconsKernel_v1(
    const       int* __restrict__ mat,
           scalar_t* __restrict__ res,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int height,
    int width
) {
  int b = blockIdx.z;
  int h = BLOCKHEIGHT4 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;
  int n_rows = h * 8 + b;
  int n_cols = w;
  scalar_t scale = scales[w];
  scalar_t zero = zeros[w];
  int i = width * h + width * (b / 8) + w;
  int shift = b % 8 * 4;
  unsigned int tmp = as_unsigned(mat[i]);
  scalar_t result = (scale * scalar_t((tmp >> shift) & 0xF) - zero);
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    scales.type(), "vecquant4recons_v1_cuda", ([&] {
      VecQuant4ReconsKernel_v1<<<blocks, threads>>>(
        mat.data<int>(), res.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        height, width
      );
    })
  );
}
