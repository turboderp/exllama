#include "rep_penalty.h"
#include <cstdlib>
#include <cstring>

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
    int beg = seq_len - s - decay;
    if (beg < 0) beg = 0;

    for (int i = 0; i < vocab_size; i++) rep_mask[i] = 1.0f;

    for (int i = seq_len; i > beg;)
    {
        uint64_t t = sequence[--i];
        if (v > rep_mask[t]) rep_mask[t] = v;
        if (--s < 0) v += dv;
    }
}

bool* g_rep_mask = NULL;
int g_vocab_size = 0;

void apply_rep_penalty_cpu
(
    const int vocab_size,
    const uint64_t* sequence,
    const float penalty_max,
    const int sustain,
    const int decay,
    const int seq_len,
    float* logits
)
{
    if (vocab_size != g_vocab_size)
    {
        if (g_rep_mask) free(g_rep_mask);
        g_vocab_size = vocab_size;
        g_rep_mask = (bool*) malloc(g_vocab_size * sizeof(bool));
    }

    memset(g_rep_mask, 0, g_vocab_size * sizeof(bool));

    float v = penalty_max;
    float dv = decay ? (1.0f - penalty_max) / (float) decay : 0.0f;

    int s = sustain == -1 ? seq_len : sustain;
    int beg = seq_len - s - decay;
    if (beg < 0) beg = 0;

    for (int i = seq_len; i > beg;)
    {
        uint64_t t = sequence[--i];
        if (!g_rep_mask[t])
        {
            if (logits[t] > 0.0) logits[t] /= v;
            else logits[t] *= v;
            g_rep_mask[t] = true;
        }
        if (--s < 0) v += dv;
    }
}