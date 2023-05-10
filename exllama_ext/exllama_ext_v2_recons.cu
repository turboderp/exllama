// #include <torch/all.h>
// #include <torch/python.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
#include <torch/types.h>
#include <torch/extension.h>

const int BLOCKWIDTH = 256;
const int BLOCKHEIGHT4 = 32;

__device__ inline int as_int(int i)
{
  return *reinterpret_cast<int*>(&i);
}

__device__ inline unsigned int as_unsigned(int i)
{
  return *reinterpret_cast<unsigned int*>(&i);
}

__global__ void VecQuant4ReconsKernel_v2
(
    const int* __restrict__ mat,
    c10::Half* __restrict__ res,
    const c10::Half* __restrict__ scales,
    const int* __restrict__ zeros,
    const int groupsize,
    int height,
    int width,
    int zero_width
)
{
    int b0 = blockIdx.z;
    int b1 = blockIdx.z << 3;  // Assume groupsize >= 8

    int h = BLOCKHEIGHT4 * blockIdx.x;
    int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    int i = width * (h + b0) + w;
    unsigned int tmp = as_unsigned(mat[i]);

    int n_cols = w;
    int n_rows = (h << 3) + b1;
    int res_i = n_rows * width + n_cols;

    int z_rows = n_rows / groupsize;
    int z_cols = n_cols >> 3;
    int z_shift = (n_cols & 0x07) << 2;
    half scale = scales[z_rows * width + n_cols];
    float scale_f = __half2float(scale);
    float zero = scale_f * (((as_unsigned(zeros[z_rows * zero_width + z_cols]) >> z_shift) & 0x0f) + 1);

    res[res_i] = __float2half(scale_f * ((tmp      ) & 0x0f) - zero); res_i += width;
    res[res_i] = __float2half(scale_f * ((tmp >>  4) & 0x0f) - zero); res_i += width;
    res[res_i] = __float2half(scale_f * ((tmp >>  8) & 0x0f) - zero); res_i += width;
    res[res_i] = __float2half(scale_f * ((tmp >> 12) & 0x0f) - zero); res_i += width;
    res[res_i] = __float2half(scale_f * ((tmp >> 16) & 0x0f) - zero); res_i += width;
    res[res_i] = __float2half(scale_f * ((tmp >> 20) & 0x0f) - zero); res_i += width;
    res[res_i] = __float2half(scale_f * ((tmp >> 24) & 0x0f) - zero); res_i += width;
    res[res_i] = __float2half(scale_f * ((tmp >> 28) & 0x0f) - zero);
}

void vecquant4recons_v2_cuda
(
    torch::Tensor mat,
    torch::Tensor res,
    torch::Tensor scales,
    torch::Tensor zeros,
    int groupsize
)
{
    int batch = BLOCKWIDTH;
    int height = mat.size(0);
    int width = mat.size(1);
    int zero_width = zeros.size(1);

    dim3 blocks
    (
        (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
        (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
        batch / 8
    );
    dim3 threads(BLOCKWIDTH);

    VecQuant4ReconsKernel_v2<<<blocks, threads>>>
    (
        mat.data_ptr<int>(),
        res.data_ptr<c10::Half>(),
        scales.data_ptr<c10::Half>(),
        zeros.data_ptr<int>(),
        groupsize,
        height,
        width,
        zero_width
    );
}