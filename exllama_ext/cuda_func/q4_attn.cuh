#ifndef _q4_attn_cuh
#define _q4_attn_cuh

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "../tuning.h"
#include "q4_matrix.cuh"

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
);

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
);

#endif