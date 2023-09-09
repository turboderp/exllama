import sys
min_version = (3, 9)
if sys.version_info < min_version:
    print("")
    print(f" ## Warning: this project requires Python {min_version[0]}.{min_version[1]} or higher.")
    print("")

import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
import cuda_ext
import json
import math
import gc
from enum import Enum

try:
    from flash_attn import flash_attn_func
except:
    pass

class ParsedEnum(Enum):

    def __str__(self):
        return self.name.lower()

    def __repr__(self):
        return str(self)

    @classmethod
    def argparse(cls, s):
        try:
            return cls[s.upper()]
        except KeyError:
            return s


class ExLlamaConfig:

    # Load config from Llama config.json

    def __init__(self, model_config_path):

        with open(model_config_path) as f:
            read_config = json.load(f)

        # Loaded/automatic settings

        self.bos_token_id = read_config["bos_token_id"] if "bos_token_id" in read_config else 1
        self.eos_token_id = read_config["eos_token_id"] if "eos_token_id" in read_config else 2
        self.pad_token_id = read_config["pad_token_id"] if "pad_token_id" in read_config else 0

        self.hidden_size = read_config["hidden_size"]
        self.initializer_range = read_config["initializer_range"]
        self.intermediate_size = read_config["intermediate_size"]
        self.num_attention_heads = read_config["num_attention_heads"]
        self.num_hidden_layers = read_config["num_hidden_layers"]
        self.rms_norm_eps = read_config["rms_norm_eps"]
        self.vocab_size = read_config["vocab_size"]

        if "num_key_value_heads" in read_config:
            self.num_key_value_heads = read_config["num_key_value_heads"]
            self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        else:
            self.num_key_value_heads = self.num_attention_heads
            self.num_key_value_groups = 1

        self.rotary_embedding_base = read_config["rope_theta"] if "rope_theta" in read_config else 10000.0
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.groupsize = None  # Autodetected
        self.act_order = False  # Autodetected
        self.empty_g_idx = False  # Autodetected

        # Required settings

        self.model_path = None  # str or list[str]
        self.device_map = ExLlamaDeviceMap(self.num_hidden_layers)

        # Optional settings

        self.max_seq_len = 2048  # Reduce to save memory. Can also be increased, ideally while also using compress_pos_emn and a compatible model/LoRA
        self.max_input_len = 2048  # Maximum length of input IDs in a single forward pass. Sequences longer than this will be processed in multiple steps
        self.max_attention_size = 2048**2  # Sequences will be processed in chunks to keep the size of the attention weights matrix <= this
        self.compress_pos_emb = 1.0  # Increase to compress positional embeddings applied to sequence
        self.alpha_value = 1.0 # Alpha value for NTK RoPE scaling. Similar to compress_pos_emb, higher values increaste ctx but add Perplexity.
        self.gpu_peer_fix = False # Apparently Torch can have problems transferring tensors directly one GPU to another sometimes. Enable this to expliticly move tensors via system RAM instead, where needed
        self.auto_map = None  # List of floats with memory allocation in GB, per CUDA device, overrides device_map

        # Tuning

        self.use_flash_attn_2 = False
        self.matmul_recons_thd = 8
        self.fused_mlp_thd = 2
        self.sdp_thd = 8
        self.fused_attn = True
        self.matmul_fused_remap = False
        self.rmsnorm_no_half2 = False
        self.rope_no_half2 = False
        self.matmul_no_half2 = False
        self.silu_no_half2 = False
        self.concurrent_streams = False

    # Copy tuning params to C++ extension

    def set_tuning_params(self):

        cuda_ext.exllama_ext.set_tuning_params(self.matmul_recons_thd,
                                               self.fused_mlp_thd,
                                               self.sdp_thd,
                                               self.matmul_fused_remap,
                                               self.rmsnorm_no_half2,
                                               self.rope_no_half2,
                                               self.matmul_no_half2,
                                               self.silu_no_half2,
                                               self.concurrent_streams)

    # Parse and set list of GPU VRAM allocations

    def set_auto_map(self, map_string):

        if map_string is None: self.auto_map = None
        else: self.auto_map = [float(alloc) for alloc in map_string.split(",")]

    def calculate_rotary_embedding_base(self):
        self.rotary_embedding_base = self.rotary_embedding_base * self.alpha_value ** (self.head_dim / (self.head_dim-2))


# 4-bit linear layer implementation

class Ex4bitLinear:

    def __init__(self, config, in_features, out_features, has_bias, tensors, key):

        self.config = config
        self.key = key
        self.in_features = in_features
        self.out_features = out_features

        self.qweight = tensors[key + ".qweight"]
        self.qzeros = tensors[key + ".qzeros"]
        self.scales = tensors[key + ".scales"]
        self.g_idx = tensors[key + ".g_idx"].cpu() if key + ".g_idx" in tensors else None
        self.bias = tensors[key + ".bias"] if has_bias else None

        if self.g_idx is not None and (self.g_idx == 0).all():
            self.config.empty_g_idx = True
            self.g_idx = None

        self.device = self.qweight.device
        self.device_index = self.device.index

        self.q4 = cuda_ext.ext_make_q4(self.qweight,
                                       self.qzeros,
                                       self.scales,
                                       self.g_idx,
                                       self.device_index)

        self.height = tensors[key + ".qweight"].shape[0] * 8
        self.width = tensors[key + ".qweight"].shape[1]

        # Infer groupsize from height of qzeros

        self.groupsize = None
        if self.qzeros.shape[0] > 1:
            self.groupsize = (self.qweight.shape[0] * 8) // self.qzeros.shape[0]
            if self.config.groupsize is None:
                self.config.groupsize = self.groupsize


        # Handle act-order matrix

        if self.g_idx is not None:

            if self.groupsize is None: raise ValueError("Found group index but no groupsize. What do?")
            self.config.act_order = True


    def lora_applies(self, lora):

        if lora is None: return False
        return self.key + ".lora_A.weight" in lora.tensors


    def lora_apply(self, lora, x):

        lora_a = lora.tensors[self.key + ".lora_A.weight"]
        lora_b = lora.tensors[self.key + ".lora_B.weight"]
        out = torch.matmul(x, lora_a)
        out = torch.matmul(out, lora_b)
        # out = cuda_ext.ext_half_matmul(x, lora_a.contiguous(), cublas = True)
        # out = cuda_ext.ext_half_matmul(out, lora_b.contiguous(), cublas = True)
        return out


    def get_lora_tensors_or_meta(self, lora):

        if not self.lora_applies(lora):
            return cuda_ext.none_tensor, cuda_ext.none_tensor
        else:
            lora_a = lora.tensors[self.key + ".lora_A.weight"]
            lora_b = lora.tensors[self.key + ".lora_B.weight"]
            return lora_a, lora_b


    def forward(self, x, lora):

        if self.lora_applies(lora):
            lora_a = lora.tensors[self.key + ".lora_A.weight"]
            lora_b = lora.tensors[self.key + ".lora_B.weight"]
            out = cuda_ext.ext_q4_matmul(x, self.q4, self.width, lora_a, lora_b)
        else:
            out = cuda_ext.ext_q4_matmul(x, self.q4, self.width)

        # out = cuda_ext.ext_q4_matmul(x, self.q4, self.width)
        # if self.lora_applies(lora):
        #     out += self.lora_apply(lora, x)

        if self.bias is not None: out.add_(self.bias)
        return out


# Llama MLP

class ExLlamaMLP:

    def __init__(self, config, tensors, key):

        self.config = config

        self.gate_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.intermediate_size, False, tensors, key + ".gate_proj")
        self.up_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.intermediate_size, False, tensors, key + ".up_proj")
        self.down_proj = Ex4bitLinear(config, self.config.intermediate_size, self.config.hidden_size, False, tensors, key + ".down_proj")

        self.act_fn = nn.SiLU()

    def fused(self, x, buffer, post_attention_layernorm, lora):

        bsz, q_len, _ = x.size()

        gate_a, gate_b = self.gate_proj.get_lora_tensors_or_meta(lora)
        up_a, up_b = self.up_proj.get_lora_tensors_or_meta(lora)
        down_a, down_b = self.down_proj.get_lora_tensors_or_meta(lora)

        temp_size = 0
        if not gate_a.is_meta: temp_size = max(temp_size, bsz * q_len * gate_a.shape[1])
        if not up_a.is_meta:   temp_size = max(temp_size, bsz * q_len * up_a.shape[1])
        if not down_a.is_meta: temp_size = max(temp_size, bsz * q_len * down_a.shape[1])

        if temp_size > 0: lora_temp = torch.empty((1, temp_size), dtype = torch.float16, device = x.device)
        else: lora_temp = cuda_ext.none_tensor

        cuda_ext.exllama_ext.q4_mlp(x.view(-1, x.shape[-1]),
                                    post_attention_layernorm.weight,
                                    self.config.rms_norm_eps,
                                    self.gate_proj.q4,
                                    self.up_proj.q4,
                                    self.down_proj.q4,
                                    gate_a, gate_b,
                                    up_a, up_b,
                                    down_a, down_b,
                                    lora_temp)


    def forward(self, x, buffer, lora):

        y = self.gate_proj.forward(x, lora)
        y = self.act_fn(y)
        y *= self.up_proj.forward(x, lora)
        y = self.down_proj.forward(y, lora)

        return y


# RMS Layer norm.

class ExLlamaRMSNorm:

    def __init__(self, config, tensors, key):

        self.config = config
        self.variance_epsilon = self.config.rms_norm_eps
        self.weight = tensors[key]


    def forward(self, hidden_states, buffer):

        hidden_states = cuda_ext.ext_rms_norm(hidden_states, self.weight, self.variance_epsilon)
        return hidden_states


# Llama attention

class ExLlamaAttention:

    def __init__(self, config, tensors, key, sin, cos, index):

        self.config = config
        self.sin = sin
        self.cos = cos
        self.index = index

        self.q_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".q_proj")
        self.k_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_key_value_heads * self.config.head_dim, False, tensors, key + ".k_proj")
        self.v_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_key_value_heads * self.config.head_dim, False, tensors, key + ".v_proj")
        self.o_proj = Ex4bitLinear(config, self.config.num_attention_heads * self.config.head_dim, self.config.hidden_size, False, tensors, key + ".o_proj")


    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

        # TODO: This seems inefficient. It should be possible to broadcast in the attention matmul to avoid building
        # temporary K/V tensors like this

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1: return hidden_states

        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


    def fused(self, hidden_states, cache, buffer, input_layernorm, lora):

        bsz, q_len, _ = hidden_states.size()
        past_len = cache.current_seq_len

        # Lora tensors

        q_a, q_b = self.q_proj.get_lora_tensors_or_meta(lora)
        k_a, k_b = self.k_proj.get_lora_tensors_or_meta(lora)
        v_a, v_b = self.v_proj.get_lora_tensors_or_meta(lora)
        o_a, o_b = self.o_proj.get_lora_tensors_or_meta(lora)

        temp_size = 0
        if not q_a.is_meta: temp_size = max(temp_size, bsz * q_len * q_a.shape[1])
        if not k_a.is_meta: temp_size = max(temp_size, bsz * q_len * k_a.shape[1])
        if not v_a.is_meta: temp_size = max(temp_size, bsz * q_len * v_a.shape[1])
        if not o_a.is_meta: temp_size = max(temp_size, bsz * q_len * o_a.shape[1])
        if temp_size > 0: lora_temp = torch.empty((1, temp_size), dtype = torch.float16, device = hidden_states.device)
        else: lora_temp = cuda_ext.none_tensor

        # Project q, k, v, apply position embeddings to k and v, update cache

        query_states = torch.empty((bsz, q_len, self.config.num_attention_heads * self.config.head_dim), dtype = torch.float16, device = hidden_states.device)
        key_states = torch.empty((bsz, q_len, self.config.num_key_value_heads * self.config.head_dim), dtype = torch.float16, device = hidden_states.device)
        value_states = torch.empty((bsz, q_len, self.config.num_key_value_heads * self.config.head_dim), dtype = torch.float16, device = hidden_states.device)

        cuda_ext.exllama_ext.q4_attn(hidden_states,
                                     input_layernorm.weight,
                                     self.config.rms_norm_eps,
                                     query_states,
                                     key_states,
                                     value_states,
                                     self.q_proj.q4,
                                     self.k_proj.q4,
                                     self.v_proj.q4,
                                     self.sin,
                                     self.cos,
                                     q_len,
                                     past_len,
                                     self.config.num_attention_heads,
                                     self.config.num_key_value_heads,
                                     self.config.head_dim,
                                     cache.key_states[self.index],
                                     cache.value_states[self.index],
                                     cache.max_seq_len,
                                     q_a, q_b,
                                     k_a, k_b,
                                     v_a, v_b,
                                     lora_temp)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim)

        # Get k, v with past

        key_states = cache.key_states[self.index].narrow(2, 0, past_len + q_len).narrow(0, 0, bsz)
        value_states = cache.value_states[self.index].narrow(2, 0, past_len + q_len).narrow(0, 0, bsz)

        # Repeat K/V heads if num_key_value_headsn_kv_heads < n_heads

        query_states.transpose_(1, 2)
        key_states = self.repeat_kv(key_states, self.config.num_key_value_groups)
        value_states = self.repeat_kv(value_states, self.config.num_key_value_groups)

        # Attention
        # TODO: Figure out if we can use cublasHgemmStridedBatched() to do this matmul without reshaping. Torch uses
        # gemmStridedBatchedEx() internally, so it should be possible.

        # -- Flash Attention 2.0

        if self.config.use_flash_attn_2 and (past_len == 0 or q_len == 1):

            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            query_states = query_states.transpose(1, 2)
            attn_output = flash_attn_func(query_states, key_states, value_states, causal = (past_len == 0))

        # -- HF Transformers regular attention, faster on shorter sequences, same VRAM usage

        else:

            key_states.transpose_(2, 3)
            attn_weights = torch.matmul(query_states, key_states)
            attn_weights /= math.sqrt(self.config.head_dim)
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

        # Output projection

        cuda_ext.exllama_ext.q4_attn_2(hidden_states,
                                       attn_output,
                                       self.o_proj.q4,
                                       o_a, o_b,
                                       lora_temp)
        # return hidden_states


    def forward(self, hidden_states, cache, buffer, lora):

        bsz, q_len, _ = hidden_states.size()
        past_len = cache.current_seq_len

        # Project q, k, v, apply position embeddings to k and v

        query_states = self.q_proj.forward(hidden_states, lora)
        key_states = self.k_proj.forward(hidden_states, lora)

        cuda_ext.exllama_ext.rope_(query_states, self.sin, self.cos, past_len, self.config.num_attention_heads, self.config.head_dim)
        cuda_ext.exllama_ext.rope_(key_states, self.sin, self.cos, past_len, self.config.num_key_value_heads, self.config.head_dim)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)
        value_states = self.v_proj.forward(hidden_states, lora).view(bsz, q_len, self.config.num_key_value_heads, self.config.head_dim).transpose(1, 2)

        # Add keys and values to cache

        new_keys = cache.key_states[self.index].narrow(2, past_len, q_len).narrow(0, 0, bsz)
        new_values = cache.value_states[self.index].narrow(2, past_len, q_len).narrow(0, 0, bsz)
        new_keys.copy_(key_states)
        new_values.copy_(value_states)

        # Key/value tensors with past

        key_states = cache.key_states[self.index].narrow(2, 0, past_len + q_len).narrow(0, 0, bsz)
        value_states = cache.value_states[self.index].narrow(2, 0, past_len + q_len).narrow(0, 0, bsz)

        # Attention

        # -- Flash Attention 2.0

        if self.config.use_flash_attn_2 and (past_len == 0 or q_len == 1):

            key_states = key_states.transpose(1, 2)
            value_states = value_states.transpose(1, 2)
            query_states = query_states.transpose(1, 2)
            attn_output = flash_attn_func(query_states, key_states, value_states, causal = (past_len == 0))

        # -- HF Transformers regular attention, faster on shorter sequences, same VRAM usage

        elif self.config.sdp_thd == 0 or q_len < self.config.sdp_thd:

            key_states = self.repeat_kv(key_states, self.config.num_key_value_groups)
            value_states = self.repeat_kv(value_states, self.config.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights /= math.sqrt(self.config.head_dim)
            if buffer.attn_mask is not None: attn_weights = attn_weights + buffer.attn_mask
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2)

        # -- Scaled dot-product attention from PyTorch 2, should be comparable to xformers (?)

        else:

            # Torch's SDP attention has a built-in causal mask feature which we can use only when there is no past, i.e.
            # it can only apply a square attention mask. It saves quite a bit of VRAM but in practice Torch seems to use
            # the same amount of memory at peak anyway.
            #
            # TODO: Apparently flash attention is disabled when supplying an attention mask tensor. Figure out if this
            # is true and maybe drop SDP altogether. If causal masking in flash-attn is updated eventually there should
            # be no need for this anyway.

            key_states = self.repeat_kv(key_states, self.config.num_key_value_groups)
            value_states = self.repeat_kv(value_states, self.config.num_key_value_groups)

            if past_len > 0 or (bsz > 1 and buffer.attn_mask is not None):
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = buffer.attn_mask, is_causal = False)
            else:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = None, is_causal = True)

            attn_output = attn_output.transpose(1, 2)

        # Output projection

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj.forward(attn_output, lora)

        return attn_output


def _rows(x):
    xdp = 1
    for y in x.shape[:-1]: xdp *= y
    return xdp

class ExLlamaDecoderLayer:

    def __init__(self, config, tensors, key, index, sin, cos):

        self.config = config
        self.index = index

        self.self_attn = ExLlamaAttention(self.config, tensors, key + ".self_attn", sin, cos, self.index)
        self.mlp = ExLlamaMLP(self.config, tensors, key + ".mlp")

        self.input_layernorm = ExLlamaRMSNorm(self.config, tensors, key + ".input_layernorm.weight")
        self.post_attention_layernorm = ExLlamaRMSNorm(self.config, tensors, key + ".post_attention_layernorm.weight")


    def forward(self, hidden_states, cache, buffer, lora):

        # Self-attention

        if self.config.fused_attn and _rows(hidden_states) == 1:

            self.self_attn.fused(hidden_states, cache, buffer, self.input_layernorm, lora)

        else:

            residual = hidden_states
            hidden_states = self.input_layernorm.forward(hidden_states, buffer)
            hidden_states = self.self_attn.forward(hidden_states, cache, buffer, lora)
            hidden_states = residual + hidden_states

        # MLP

        if self.config.fused_mlp_thd > 0 and _rows(hidden_states) <= self.config.fused_mlp_thd:

            self.mlp.fused(hidden_states, buffer, self.post_attention_layernorm, lora)

        else:

            residual = hidden_states
            hidden_states = self.post_attention_layernorm.forward(hidden_states, buffer)
            hidden_states = self.mlp.forward(hidden_states, buffer, lora)
            hidden_states = residual + hidden_states

        return hidden_states


# Persistent cache for inference. Allocate the whole thing up front.

class ExLlamaCache:

    def __init__(self, model, batch_size = 1, max_seq_len = -1, copy_from = None):

        self.model = model
        self.config = self.model.config
        self.max_seq_len = max_seq_len if max_seq_len != -1 else self.config.max_seq_len
        self.batch_size = batch_size

        self.key_states = []
        self.value_states = []
        self.current_seq_len = 0

        # Preallocate full-length cache

        for i in range(self.config.num_hidden_layers):

            if copy_from is None:

                p_key_states = torch.zeros(self.batch_size, self.config.num_key_value_heads, self.max_seq_len, self.config.head_dim, dtype = torch.float16, device = self.model.config.device_map.layers[i])
                p_value_states = torch.zeros(self.batch_size, self.config.num_key_value_heads, self.max_seq_len, self.config.head_dim, dtype = torch.float16, device = self.model.config.device_map.layers[i])

            else:

                p_key_states = copy_from.key_states[i].clone()
                p_value_states = copy_from.value_states[i].clone()

            self.key_states.append(p_key_states)
            self.value_states.append(p_value_states)


    def zero(self):

        for i in range(self.config.num_hidden_layers):
            self.key_states[i].zero_()
            self.value_states[i].zero_()


    def clone(self):

        new = ExLlamaCache(self.model, batch_size = self.batch_size, max_seq_len = self.max_seq_len, copy_from = self)
        return new


    def roll_left(self):

        for i in range(self.config.num_hidden_layers):

            self.key_states[i] = torch.roll(self.key_states[i], shifts = -1, dims = 2)
            self.value_states[i] = torch.roll(self.value_states[i], shifts = -1, dims = 2)

        self.current_seq_len -= 1


    def copy_states(self, target, from_column, from_columns, to_column, to_columns, from_row, from_rows, to_row, to_rows):

        assert from_rows == 1
        assert from_columns == to_columns
        assert to_column + to_columns <= target.max_seq_len
        assert from_column + from_columns <= self.max_seq_len

        for i in range(self.config.num_hidden_layers):

            source_view_k = self.key_states[i].narrow(0, from_row, from_rows).narrow(2, from_column, from_columns)
            source_view_v = self.value_states[i].narrow(0, from_row, from_rows).narrow(2, from_column, from_columns)
            target_view_k = target.key_states[i].narrow(0, to_row, to_rows).narrow(2, to_column, to_columns)
            target_view_v = target.value_states[i].narrow(0, to_row, to_rows).narrow(2, to_column, to_columns)

            if to_rows > 1:

                source_view_k = source_view_k.expand_as(target_view_k)
                source_view_v = source_view_v.expand_as(target_view_v)

            target_view_k.copy_(source_view_k)
            target_view_v.copy_(source_view_v)


# Device map for the model.

class ExLlamaDeviceMap:

    def __init__(self, num_layers):

        self.num_layers = num_layers

        self.embed_tokens = "cpu"  # Embedding table on CPU saves 400 MB on the 30B model with no measurable impact on performance
        self.lm_head = "cuda:0"
        self.norm = "cuda:0"
        self.layers = ["cuda:0"] * self.num_layers


    def get_layers_devs(self):

        return sorted(list(set(self.layers)))


    def get_all_devs(self):

        return sorted(list(set(self.layers + [self.lm_head, self.norm, self.embed_tokens])))


    def map(self, key):

        if key.startswith("lm_head."): return self.lm_head
        if key.startswith("model.embed_tokens."): return self.embed_tokens
        if key.startswith("model.norm."): return self.norm

        if key.startswith("model.layers."):
            num = int(key.split(".")[2])
            return self.layers[num]

        raise ValueError("Unknown key: " + key)


class ExLlamaBuffer:

    config: ExLlamaConfig

    def __init__(self, config):

        self.config = config

    # Attention mask

    attn_mask: torch.Tensor = None

    # Move to device

    def to(self, device):

        new = ExLlamaBuffer(self.config)
        new.attn_mask = None if self.attn_mask is None else _move_tensor(self.attn_mask, device, "attn_mask", self.config)
        return new


def _device_to_int(device):

    return int(device[device.find(":") + 1:])

def _skip_key(key):

    if key.endswith("_proj.bias"): return True
    if key.endswith(".rotary_emb.inv_freq"): return True
    return False

def _move_tensor(tensor, new_device, name, config):
    device = str(tensor.device)
    if device == new_device: return tensor
    if config.gpu_peer_fix:
        if str(device).startswith("cuda:") and str(new_device).startswith("cuda:"):
            tensor = tensor.to("cpu")
    return tensor.to(new_device)

def _layer_dtype_size(key):
    if key.endswith(".weight"): return 2
    if key.endswith(".qweight"): return 4
    if key.endswith(".qzeros"): return 4
    if key.endswith(".scales"): return 2
    if key.endswith(".g_idx"): return 0
    raise ValueError("Unrecognized layer: " + key)


class ExLlama:

    def __init__(self, config):

        self.config = config

        # Copy tuning parameters to C++ extension

        self.config.set_tuning_params()

        # Read tensor list from file(s)

        if isinstance(self.config.model_path, str): model_path = [self.config.model_path]
        else: model_path = self.config.model_path

        # Read tensor list from file(s), and measure layer sizes

        load_keys = {}

        decoder_size = 0
        norm_size = 0
        head_size = 0

        for path in model_path:
            with safe_open(path, framework = "pt", device = "cpu") as f:
                for key in f.keys():

                    if _skip_key(key): continue

                    load_keys[key] = path

                    if key.startswith("model.layers.0."):
                        tensor_slice = f.get_slice(key)
                        shape = tensor_slice.get_shape()
                        decoder_size += math.prod(shape) * _layer_dtype_size(key)
                        del tensor_slice

                    if key.startswith("model.norm."):
                        tensor_slice = f.get_slice(key)
                        shape = tensor_slice.get_shape()
                        norm_size += math.prod(shape) * _layer_dtype_size(key)
                        del tensor_slice

                    if key.startswith("lm_head."):
                        tensor_slice = f.get_slice(key)
                        shape = tensor_slice.get_shape()
                        head_size += math.prod(shape) * _layer_dtype_size(key)
                        del tensor_slice

        # Begin auto mapping if enabled

        if self.config.auto_map is not None:

            self.config.device_map.embed_tokens = "cpu"
            self.config.device_map.layers = ["cuda:0"] + ["?"] * (self.config.num_hidden_layers - 1)

            # Assign layers automatically

            device_usage = 0
            device_index = 0
            layer_index_device = 0
            max_usage = self.config.auto_map[device_index] * (1024 ** 3)

            for layer in range(self.config.num_hidden_layers + 2):

                this_layer_size = decoder_size
                if layer == self.config.num_hidden_layers + 0: this_layer_size = norm_size
                elif layer == self.config.num_hidden_layers + 1: this_layer_size = head_size

                while device_usage + this_layer_size > max_usage:
                    device_index += 1
                    device_usage = 0
                    layer_index_device = 0
                    max_usage = self.config.auto_map[device_index] * (1024 ** 3)
                    if device_index >= len(self.config.auto_map): raise ValueError("Model too large for device allocation scheme.")

                target = f"cuda:{device_index}"
                if layer == self.config.num_hidden_layers + 0: self.config.device_map.norm = target
                elif layer == self.config.num_hidden_layers + 1: self.config.device_map.lm_head = target
                else: self.config.device_map.layers[layer] = f"cuda:{device_index}"

                device_usage += this_layer_size
                layer_index_device += 1

         # Load up to 1 GB of tensors at a time, closing and reopening the file in between each chunk

        max_dq_buffer_size = 0
        tensors = {}

        st_mem = 0
        MAX_ST_MEM = 1024**3
        f = None
        prev_path = ""
        for key, path in load_keys.items():

            device = self.config.device_map.map(key)

            if f is None or st_mem > MAX_ST_MEM or path != prev_path:
                if f is not None: del f
                f = safe_open(path, framework = "pt", device = "cpu")
                prev_path = path
                st_mem = 0

            tensor = f.get_tensor(key)
            size = tensor.numel() * tensor.element_size()
            st_mem += size

            if key.endswith(".scales"): tensor = tensor.half()
            if key == "lm_head.weight": tensor = tensor.float() if device == "cpu" else tensor.half()
            if key == "model.norm.weight": tensor = tensor.half()
            if key.endswith(".embed_tokens.weight"): tensor = tensor.half()
            if key.endswith(".input_layernorm.weight"): tensor = tensor.half()
            if key.endswith(".post_attention_layernorm.weight"): tensor = tensor.half()

            if device == "cpu": keep_tensor = tensor.clone()
            else: keep_tensor = tensor.to(device)
            del tensor

            if key.endswith(".qweight"): max_dq_buffer_size = max(max_dq_buffer_size, keep_tensor.numel() * 8)

            tensors[key] = keep_tensor

        del f

        # Head

        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False, device = "meta")
        self.lm_head.weight = nn.Parameter(tensors["lm_head.weight"])
        # self.lm_head_data = tensors["lm_head.weight"].transpose(0, 1).contiguous()

        # Token embeddings

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id, device = "meta")
        self.embed_tokens.weight = nn.Parameter(tensors["model.embed_tokens.weight"])
        with torch.no_grad():
            self.embed_tokens.weight[self.config.pad_token_id] = 0

        # Norm

        self.norm = ExLlamaRMSNorm(self.config, tensors, "model.norm.weight")

        # Prepare position embeddings for max seq length

        devs = self.config.device_map.get_layers_devs()

        self.sincos = {}
        for device in devs:

            inv_freq = 1.0 / (self.config.rotary_embedding_base ** (torch.arange(0, self.config.head_dim, 2, device = device).float() / self.config.head_dim))
            t = torch.arange(self.config.max_seq_len, device = device, dtype = torch.float32)
            if self.config.compress_pos_emb != 1.0: t /= self.config.compress_pos_emb

            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)

            sin = emb.sin()[None, None, :, :].half()
            cos = emb.cos()[None, None, :, :].half()

            self.sincos[device] = (sin, cos)

        # Decoder layers

        modules = []
        device_layer_index = [0] * len(devs)

        for i in range(self.config.num_hidden_layers):

            device = self.config.device_map.layers[i]
            sin, cos = self.sincos[device]

            layer = ExLlamaDecoderLayer(self.config, tensors, f"model.layers.{i}", i, sin, cos)

            modules.append(layer)

        self.layers = modules

        # Prepare CUDA buffers

        self.buffers = []
        for dev in self.config.device_map.get_layers_devs():

            device_buffers = {}
            self.buffers.append(device_buffers)

            temp_state = torch.zeros((config.max_input_len, config.intermediate_size), dtype = torch.float16, device = dev)
            temp_mlp = torch.zeros((config.fused_mlp_thd * 2, config.intermediate_size), dtype = torch.float16, device = dev)
            temp_zeros_float = torch.zeros((1, 65536), dtype = torch.float32, device = dev)
            temp_dq = torch.zeros((1, max_dq_buffer_size), dtype = torch.float16, device = dev)

            device_buffers["temp_state"] = temp_state
            device_buffers["temp_mlp"] = temp_mlp
            device_buffers["temp_zeros_float"] = temp_zeros_float
            device_buffers["temp_dq"] = temp_dq

            cuda_ext.exllama_ext.prepare_buffers(torch.device(dev),
                                                 temp_state,
                                                 temp_mlp,
                                                 temp_zeros_float,
                                                 temp_dq)

        # Clear the cache

        torch.cuda.empty_cache()


    def forward(self,
                input_ids,
                cache,
                last_id_only = True,
                preprocess_only = False,
                lora = None,
                output_device = None,
                input_mask = None):

        q_len = input_ids.shape[-1]
        remaining_q_len = q_len
        bsz = input_ids.shape[0]

        assert input_mask is None or (input_mask.shape[-1] >= input_ids.shape[-1] and input_mask.shape[-2] == input_ids.shape[-2])

        # The buffers can only fit max_input_len tokens, so with larger batch sizes we reduce our work size correspondingly.

        effective_max_input_len = self.config.max_input_len // bsz

        # Split sequence

        result = None

        chunk_begin = 0
        while chunk_begin < q_len:

            # Limit chunk_size to max_input_len

            chunk_size = min(remaining_q_len, effective_max_input_len)

            # Limit chunk_size to keep size of attention operation <= max_attention_size, unless using flash-attn

            if not self.config.use_flash_attn_2 or chunk_begin > 0:

                past_len = cache.current_seq_len
                attn_size = (past_len + remaining_q_len) * remaining_q_len
                max_a = self.config.max_attention_size
                if attn_size > max_a:
                    cs = (math.sqrt(past_len ** 2 + 4 * max_a) - past_len) / 2
                    chunk_size = min(chunk_size, math.floor(cs))

            # Process chunk

            chunk_end = min(chunk_begin + chunk_size, q_len)

            _last_id_only = last_id_only
            _preprocess_only = preprocess_only or (chunk_end < q_len and last_id_only)

            r = self._forward(input_ids[:, chunk_begin : chunk_end],
                             cache,
                             _last_id_only,
                             _preprocess_only,
                             lora,
                             output_device,
                             input_mask)

            if not _preprocess_only:
                result = r if result is None else torch.cat((result, r), dim = 1)

            chunk_begin = chunk_end
            remaining_q_len -= chunk_size

        return result


    def _forward(self,
                 input_ids,
                 cache,
                 last_id_only = True,
                 preprocess_only = False,
                 lora = None,
                 output_device = None,
                 input_mask = None):

        # if torch.is_grad_enabled():
        #     raise ValueError("Forward pass called with gradients enabled. Back propagation is not supported yet.")
        with torch.no_grad():

            batch_size, seq_len = input_ids.shape
            past_len = cache.current_seq_len
            if output_device is None: output_device = input_ids.device

            buffer = ExLlamaBuffer(self.config)

            # Build attention mask on first device, copy to others if necessary

            devs = self.config.device_map.get_layers_devs()

            # if not self.config.use_flash_attn_2:

            if seq_len > 1 or input_mask is not None:

                attn_mask = torch.zeros(batch_size, 1, seq_len, past_len + seq_len, dtype = torch.float16, device = devs[0])
                attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), -65504.))
                attn_mask[:, :, : seq_len - 1, past_len + 1: past_len + seq_len] = attn_mask_triu

                if input_mask is not None:

                    input_mask = input_mask[:, :past_len + seq_len]
                    input_mask = _move_tensor(input_mask, devs[0], "input_mask", self.config)
                    input_mask = torch.where(input_mask, 0, -65504.).half()
                    input_mask = input_mask.unsqueeze(1).unsqueeze(2)
                    attn_mask = torch.minimum(attn_mask, input_mask)

            else:

                attn_mask = None
                # attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len + past_len, dtype = torch.float16, device = devs[0])

            buffer.attn_mask = attn_mask

            # else:
            #
            #     buffer.attn_mask = None

            # Embeddings
            # TODO: Allow passing input embeddings instead of IDs

            input_ids = _move_tensor(input_ids, self.config.device_map.embed_tokens, "input_ids", self.config)
            hidden_states = self.embed_tokens(input_ids)

            # Split buffers to devices

            buffers = {devs[0]: buffer}
            for device in devs[1:]:
                buffers[device] = buffer.to(device)

            # Decoder layers

            for i, decoder_layer in enumerate(self.layers):

                device = self.config.device_map.layers[i]
                hidden_states = _move_tensor(hidden_states, device, "hidden_states", self.config)

                hidden_states = decoder_layer.forward(hidden_states, cache, buffers[device], lora)

            cache.current_seq_len += seq_len

            # Early exit when we don't need logits

            if preprocess_only: return None

            # Norm

            hidden_states = _move_tensor(hidden_states, self.config.device_map.norm, "hidden_states", self.config)
            hidden_states = self.norm.forward(hidden_states, buffer)

            # Head

            if last_id_only: hidden_states = hidden_states[:, -1:, :].contiguous()
            if self.config.device_map.lm_head == "cpu": hidden_states = hidden_states.float()

            hidden_states = _move_tensor(hidden_states, self.config.device_map.lm_head, "hidden_states", self.config)
            logits = self.lm_head(hidden_states)
            # logits = cuda_ext.matmul_half(hidden_states, self.lm_head_data, cublas = False)

            logits = logits.float()
            logits = _move_tensor(logits, output_device, "logits", self.config)
            return logits


    # Free unmanaged resources allocated by the C++ extension. Call this before dereferencing the ExLlama object,
    # e.g. if you intend to create a new instance to load another model, but don't call it in a destructor that wraps
    # the object, since it relies on CUDA function calls and the CUDA context is one of the first things to go when
    # a PyTorch application terminates, before other managed objects are destroyed.

    def free_unmanaged(self):

        cuda_ext.exllama_ext.cleanup()
