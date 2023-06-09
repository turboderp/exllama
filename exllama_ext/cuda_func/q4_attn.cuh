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
    const int q_len,
    const int dim,
    const int head_dim,
    const int num_heads,
    const int past_len,
    half* key_cache,
    half* value_cache,
    const int max_seq_len,
    const int device_index
);

void q4_attn_2_cuda
(
    ExLlamaTuning* tuningParams,
    half* x,
    half* attn_output,
    Q4Matrix* o_proj,
    const int height
);

#endif