#include "q4_mlp.cuh"
#include "q4_matmul.cuh"
#include "rope.cuh"
#include "rms_norm.cuh"
#include "half_matmul.cuh"
#include "../cuda_buffers.cuh"
#include "../util.cuh"
#include "../matrix.cuh"
#if defined(USE_ROCM)
#include "../hip_compat.cuh"
#endif

const int THREADS_X = 32;
const int THREADS_Y = 1;
const int THREADS_Z = 4;
const int BLOCKSIZE_X = 2; // 2*half == 1*uint32_t
const int BLOCKSIZE_Z = 4; // num_heads must be divisible by BLOCKSIZE_Z  TODO: Check that this is the case when Llama2-34b releases

__global__ void update_cache_kernel
(
    const half* __restrict__ key_states,
    const half* __restrict__ value_states,
    half* __restrict__ key_cache,
    half* __restrict__ value_cache,
    const int head_dim,
    const int num_kv_heads,
    const int q_len,
    const int max_seq_len,
    const int past_len
)
{
    //int state_shape[]  = {              num_kv_heads,                     q_len, head_dim };
    int state_stride[] = {                  head_dim,   head_dim * num_kv_heads,        1 };
    int state_pos[]    = {                         0,                         0,        0 };

    //int cache_shape[]  = {              num_kv_heads,               max_seq_len, head_dim };
    int cache_stride[] = {    max_seq_len * head_dim,                  head_dim,        1 };
    int cache_pos[]    = {                         0,                  past_len,        0 };

    int size[]         = {              num_kv_heads,                  q_len, head_dim };

    int x = (blockIdx.x * THREADS_X + threadIdx.x) * BLOCKSIZE_X; 
    int y = blockIdx.y * THREADS_Y + threadIdx.y;
    int z = (blockIdx.z * THREADS_Z + threadIdx.z) * BLOCKSIZE_Z;
    
    if (x >= size[2]) return;
    if (y >= size[1]) return;
    if (z >= size[0]) return;

    int state_offset = (z + state_pos[0]) * state_stride[0] + (y + state_pos[1]) * state_stride[1] + (x + state_pos[2]) * state_stride[2];
    int cache_offset = (z + cache_pos[0]) * cache_stride[0] + (y + cache_pos[1]) * cache_stride[1] + (x + cache_pos[2]) * cache_stride[2];

    const uint32_t* key_ptr = (uint32_t*) (key_states + state_offset);
    const uint32_t* value_ptr = (uint32_t*) (value_states + state_offset);
    uint32_t* key_cache_ptr = (uint32_t*) (key_cache + cache_offset);
    uint32_t* value_cache_ptr = (uint32_t*) (value_cache + cache_offset);

    #pragma unroll
    for (int k = 0; k < BLOCKSIZE_Z; k++)
    {
        *key_cache_ptr = *key_ptr;
        key_ptr += state_stride[0] / BLOCKSIZE_X;
        key_cache_ptr += cache_stride[0] / BLOCKSIZE_X;
    }
    #pragma unroll
    for (int k = 0; k < BLOCKSIZE_Z; k++)
    {
        *value_cache_ptr = *value_ptr;
        value_ptr += state_stride[0] / BLOCKSIZE_X;
        value_cache_ptr += cache_stride[0] / BLOCKSIZE_X;
    }
}

void q4_attn_cuda
(
    ExLlamaTuning* tuningParams,
    cudaStream_t stream,
    cublasHandle_t handle,
    half* x,
    const half* rms_norm_weight,    // shape == (x.shape[1],) == (dim,)
    float epsilon,
    half* query_states,
    half* key_states,
    half* value_states,
    Q4Matrix* q_proj,
    Q4Matrix* k_proj,
    Q4Matrix* v_proj,
    half* sin,
    half* cos,
    const int bsz,
    const int q_len,
    const int dim,
    const int head_dim,
    const int num_heads,
    const int num_kv_heads,
    const int past_len,
    half* key_cache,
    half* value_cache,
    const half* q_a,
    const half* q_b,
    const int q_rank,
    const half* k_a,
    const half* k_b,
    const int k_rank,
    const half* v_a,
    const half* v_b,
    const int v_rank,
    half* lora_temp,
    const int max_seq_len,
    const int device_index
)
{
    // Cache update grid

    dim3 threads(THREADS_X, THREADS_Y, THREADS_Z);

    dim3 blocks
    (
        ((head_dim + THREADS_X - 1) / THREADS_X + BLOCKSIZE_X - 1) / BLOCKSIZE_X,
        q_len,
        ((num_kv_heads + THREADS_Z - 1) / THREADS_Z + BLOCKSIZE_Z - 1) / BLOCKSIZE_Z
    );

    int _rows_per_batch = q_len * num_heads;
    int _rows_per_batch_kv = q_len * num_kv_heads;

    CudaBuffers* buffers = get_buffers(device_index);

    // Layernorm

    half* temp_x = buffers->temp_state + q_len * dim;
    rms_norm_cuda(tuningParams, x, rms_norm_weight, temp_x, epsilon, q_len, dim, device_index);

    // Adapters

    if (q_a)
    {
        half_matmul_cublas_cuda(tuningParams, temp_x, q_a, lora_temp, q_len, dim, q_rank, handle);
        half_matmul_cublas_cuda(tuningParams, lora_temp, q_b, query_states, q_len, q_rank, dim, handle);
    }
    if (k_a)
    {
        half_matmul_cublas_cuda(tuningParams, temp_x, k_a, lora_temp, q_len, dim, k_rank, handle);
        half_matmul_cublas_cuda(tuningParams, lora_temp, k_b, key_states, q_len, k_rank, dim, handle);
    }
    if (v_a)
    {
        half_matmul_cublas_cuda(tuningParams, temp_x, v_a, lora_temp, q_len, dim, v_rank, handle);
        half_matmul_cublas_cuda(tuningParams, lora_temp, v_b, value_states, q_len, v_rank, dim, handle);
    }

    if (!tuningParams->concurrent_streams)
    {
        // Project q, k, v

        q4_matmul_cuda(tuningParams, temp_x, q_len, q_proj, query_states, q_a ? true : false);
        q4_matmul_cuda(tuningParams, temp_x, q_len, k_proj, key_states, k_a ? true : false);
        q4_matmul_cuda(tuningParams, temp_x, q_len, v_proj, value_states, v_a ? true : false);

        // Positional embeddings q, k

        rope_cuda(tuningParams, query_states, sin, cos, bsz, _rows_per_batch, head_dim, num_heads, past_len);
        rope_cuda(tuningParams, key_states, sin, cos, bsz, _rows_per_batch_kv, head_dim, num_kv_heads, past_len);

        // Update cache tensors with projected k, v

        update_cache_kernel<<<blocks, threads>>>(key_states, value_states, key_cache, value_cache, head_dim, num_kv_heads, q_len, max_seq_len, past_len);
    }
    else
    {
        // Project q, k, v, add positional embeddings to q, k, update cache tensors with projected k, v

        cudaStream_t str_1 = buffers->alt_stream_1;
        cudaStream_t str_2 = buffers->alt_stream_2;
        cudaStream_t str_3 = buffers->alt_stream_3;
        cudaEvent_t sync_1 = buffers->alt_stream_1_done;
        cudaEvent_t sync_2 = buffers->alt_stream_2_done;
        cudaEvent_t sync_3 = buffers->alt_stream_3_done;

        // str_1: project q, positions q, sync

        q4_matmul_cuda(tuningParams, temp_x, q_len, q_proj, query_states, q_a ? true : false, str_1);
        rope_cuda(tuningParams, query_states, sin, cos,  bsz, _rows_per_batch, head_dim, num_kv_heads, past_len, str_1);
        cudaEventRecord(sync_1, str_1);

        // str_2: project k, positions k, sync

        q4_matmul_cuda(tuningParams, temp_x, q_len, k_proj, key_states, k_a ? true : false, str_2);
        rope_cuda(tuningParams, key_states, sin, cos,  bsz, _rows_per_batch_kv, head_dim, num_kv_heads, past_len, str_2);
        cudaEventRecord(sync_2, str_2);

        // str_3: project v, wait for str_2, copy (k,v) to cache, sync

        q4_matmul_cuda(tuningParams, temp_x, q_len, v_proj, value_states, v_a ? true : false, buffers->alt_stream_3);
        cudaStreamWaitEvent(str_3, sync_2, 0);
        update_cache_kernel<<<blocks, threads, 0, str_3>>>(key_states, value_states, key_cache, value_cache, head_dim, num_kv_heads, q_len, max_seq_len, past_len);
        cudaEventRecord(sync_3, str_3);

        // default: wait for str_1 and str_3

        cudaStreamWaitEvent(NULL, sync_1, 0);
        cudaStreamWaitEvent(NULL, sync_3, 0);
    }
}

void q4_attn_2_cuda
(
    ExLlamaTuning* tuningParams,
    cublasHandle_t handle,
    half* x,
    half* attn_output,
    Q4Matrix* o_proj,
    const int height,
    const half* o_a,
    const half* o_b,
    const int o_rank,
    half* lora_temp
)
{
    if (o_a)
    {
        int dim = o_proj->height;
        half_matmul_cublas_cuda(tuningParams, attn_output, o_a, lora_temp, height, dim, o_rank, handle);
        half_matmul_cublas_cuda(tuningParams, lora_temp, o_b, x, height, o_rank, dim, handle, true);
    }

    q4_matmul_cuda(tuningParams, attn_output, height, o_proj, x, true);
}
