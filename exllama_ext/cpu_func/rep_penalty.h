#ifndef _rep_penalty_h
#define _rep_penalty_h

#include <cstdint>
#include <cstdio>

void rep_penalty_cpu
(
    const int vocab_size,
    const uint64_t* sequence,
    float* rep_mask,
    const float penalty_max,
    const int sustain,
    const int decay,
    const int seq_len
);

void apply_rep_penalty_cpu
(
    const int vocab_size,
    const uint64_t* sequence,
    const float penalty_max,
    const int sustain,
    const int decay,
    const int seq_len,
    float* logits
);


#endif
