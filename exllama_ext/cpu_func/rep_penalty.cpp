#include "rep_penalty.h"

void rep_penalty_cpu
(
    const int vocab_size,
    const uint64_t* sequence,
    float* rep_mask,
    const float penalty_max,
    const int sustain,
    const int decay,
    const int seq_len
)
{
    float v = penalty_max;
    float dv = decay ? (1.0f - penalty_max) / (float) decay : 0.0f;

    int s = sustain == -1 ? seq_len : sustain;
    int beg = seq_len - sustain - decay;
    if (beg < 0) beg = 0;

    for (int i = 0; i < vocab_size; i++) rep_mask[i] = 1.0f;

    for (int i = seq_len; i > beg;)
    {
        uint64_t t = sequence[--i];
        if (v > rep_mask[t]) rep_mask[t] = v;
        if (--s < 0) v += dv;
    }
}