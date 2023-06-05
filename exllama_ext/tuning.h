#ifndef _tuning_h
#define _tuning_h

struct ExLlamaTuning
{
    int matmul_recons_thd;
    int fused_mlp_thd;
    int sdp_thd;

    bool rmsnorm_no_half2;
    bool rope_no_half2;
};

#endif