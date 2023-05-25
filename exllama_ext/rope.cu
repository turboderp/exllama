#include "rope.h"
#include "util.h"
#include "matrix.h"

const int THREADS_X = 32;
const int THREADS_Y = 4;
const int MAX_POS_EMBEDDINGS = 32768;  // Actual number doesn't matter

__global__ void rope_cuda_kernel
(
    half* x,
    const half* sin,
    const half* cos,
    int rows,
    int head_dim,
    int num_heads,
    int past_len
)
{
    MatrixView_half_rw x_(x, rows, head_dim);
    MatrixView_half sin_(sin, MAX_POS_EMBEDDINGS, head_dim);
    MatrixView_half cos_(cos, MAX_POS_EMBEDDINGS, head_dim);

    // Assume head_dim is a power of two (it's always 128 for Llama)

    int column = (blockIdx.x * THREADS_X + threadIdx.x) * 2;
    int row = blockIdx.y * THREADS_Y + threadIdx.y;
    if (row >= rows) return;

    // Get sin and cos

    int sincos_row = past_len + row / num_heads;
    int half_dim = head_dim / 2;

    half2 cos2_l = cos_.item_half2(sincos_row, column);
    half2 cos2_r = cos_.item_half2(sincos_row, column + half_dim);
    half2 sin2_l = sin_.item_half2(sincos_row, column);
    half2 sin2_r = sin_.item_half2(sincos_row, column + half_dim);
    sin2_l = __hneg2(sin2_l);

    // Apply embedding to num_heads rows

    //#pragma unroll
    //for (int k = row; k < row + num_heads; k++)
    int k = row;
    {
//         if (k == 0)
//         {
//             printf("%f %f %f %f  - ", __half2float(sin2_l.x), __half2float(sin2_l.y), __half2float(sin2_r.x), __half2float(sin2_r.y));
//             printf("%f %f %f %f \n", __half2float(cos2_l.x), __half2float(cos2_l.y), __half2float(cos2_r.x), __half2float(cos2_r.y));
//         }

        half2 item2_l = x_.item_half2(k, column);
        half2 item2_r = x_.item_half2(k, column + half_dim);
        half2 item2_ls = __hmul2(item2_r, sin2_l);
        half2 item2_rs = __hmul2(item2_l, sin2_r);
        item2_l = __hfma2(item2_l, cos2_l, item2_ls);
        item2_r = __hfma2(item2_r, cos2_r, item2_rs);
//         item2_l = __hmul2(item2_l, cos2_l);
//         item2_r = __hmul2(item2_r, cos2_r);
//         item2_l = __hadd2(item2_l, item2_ls);
//         item2_r = __hadd2(item2_r, item2_rs);
        x_.set_half2(k, column, item2_l);
        x_.set_half2(k, column + half_dim, item2_r);
//        x_.set_half2(k, column, __half2half2(__int2half_rn(69)));
//        x_.set_half2(k, column + half_dim, __half2half2(__int2half_rn(69)));

    }
}

cudaError_t rope_cuda
(
    half* x,
    const half* sin,
    const half* cos,
    const int rows,
    const int head_dim,
    const int num_heads,
    const int past_len
)
{

    dim3 threads(THREADS_X, THREADS_Y, 1);

    dim3 blocks
    (
        (head_dim + THREADS_X - 1) / THREADS_X / 2 / 2,
        (rows + THREADS_Y - 1) / THREADS_Y, //(rows + num_heads - 1) / num_heads,
        1
    );

    rope_cuda_kernel<<<blocks, threads>>>(x, sin, cos, rows, head_dim, num_heads, past_len);

}
