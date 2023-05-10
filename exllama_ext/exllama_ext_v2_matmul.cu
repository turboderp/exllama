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

__global__ void VecQuant4MatMulKernel_v2
(
    const half2* __restrict__ vec,
    const int* __restrict__ mat,
    c10::Half* __restrict__ mul,
    const c10::Half* __restrict__ scales,
    const int* __restrict__ zeros,
    int groupsize,
	int batch,
	int vec_height,
    int height,
    int width,
    int zero_width
)
{
    const int blockwidth2 = BLOCKWIDTH >> 1;

    int b = blockIdx.z;
    int h = BLOCKHEIGHT4 * blockIdx.x;
    int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    __shared__ half2 blockvec[blockwidth2];
    if (threadIdx.x < blockwidth2)
        blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * blockwidth2 + threadIdx.x];

    int i = width * h + w;
    int g_h = h * 8;
    int k = 0;

    int z_w = w >> 3;
    int z_mod = (w & 0x7) << 2;

    half2 res_acc = {};

    unsigned int tmp;

    half* scales_half = (half*)scales;

    __syncthreads();

    #pragma unroll
    while (k < blockwidth2)
    {
        int g = (g_h + (k << 1)) / groupsize;
        half scale_half = scales_half[g * width + w];
        half scale_half_neg = __hmul(scale_half, __int2half_rn(-1));
        half2 scale_half2 = __half2half2(scale_half);
        half2 zero = __half2half2(__hmul(scale_half_neg, __int2half_rn(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0x0f) + 1)));

        tmp = as_unsigned(mat[i]);
        unsigned char val1, val2;
        half2 deq2_val;

        res_acc = __hfma2(__hfma2(__halves2half2(__int2half_rn((tmp      ) & 0x0f), __int2half_rn((tmp >>  4) & 0x0f)), scale_half2, zero), blockvec[k++], res_acc);
        res_acc = __hfma2(__hfma2(__halves2half2(__int2half_rn((tmp >>  8) & 0x0f), __int2half_rn((tmp >> 12) & 0x0f)), scale_half2, zero), blockvec[k++], res_acc);
        res_acc = __hfma2(__hfma2(__halves2half2(__int2half_rn((tmp >> 16) & 0x0f), __int2half_rn((tmp >> 20) & 0x0f)), scale_half2, zero), blockvec[k++], res_acc);
        res_acc = __hfma2(__hfma2(__halves2half2(__int2half_rn((tmp >> 24) & 0x0f), __int2half_rn((tmp >> 28) & 0x0f)), scale_half2, zero), blockvec[k++], res_acc);

        i += width;
    }

    half res = __hadd(res_acc.x, res_acc.y);
    half* mul2 = (half*)mul;
    atomicAdd(&mul2[b * width + w], res);
}

void vecquant4matmul_v2_cuda
(
    torch::Tensor vec,
    torch::Tensor mat,
    torch::Tensor mul,
    torch::Tensor scales,
    torch::Tensor zeros,
    int groupsize,
    int vec_height
)
{
    int batch = vec.size(0);
    int height = mat.size(0);
    int width = mat.size(1);
    int zero_width = zeros.size(1);

    dim3 blocks
    (
        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
        batch
    );
    dim3 threads(BLOCKWIDTH);

    VecQuant4MatMulKernel_v2<<<blocks, threads>>>
    (
        (half2*)vec.data_ptr(),
        mat.data_ptr<int>(),
        mul.data_ptr<c10::Half>(),
        scales.data_ptr<c10::Half>(),
        zeros.data_ptr<int>(),
        groupsize,
        batch, vec_height, height, width, zero_width
    );
}
