import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
import cuda_ext
import json
import math
from enum import Enum
import threading
import sys
import struct
from typing import List

# Magic numbers

optimal_switch_thd = 6  # Model mostly runs one token at a time, or many. So this probably doesn't matter too much.


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

    class AttentionMethod(ParsedEnum):

        PYTORCH_MATMUL = 1  # Regular attention from HF implementation. Dog poop.
        PYTORCH_SCALED_DP = 2  # Seems more memory-efficient than xformers


    class MatmulMethod(ParsedEnum):

        QUANT_ONLY = 1  # Use the quantized matmul
        SWITCHED = 2  # Switch between quantized matmul and FP16 reconstruction (best)
        PYTORCH_ONLY = 3  # Always reconstruct and perform FP16 matmul


    class MLPMethod(ParsedEnum):

        NORMAL = 1  # Regular MLP as in LlamaModel (best)
        SWITCHED = 2  # Switch between normal and fused MLP
        FUSED = 3  # Always use fused MLP


    # Load config from Llama config.json

    def __init__(self, model_config_path):

        with open(model_config_path) as f:
            read_config = json.load(f)

        # Loaded/automatic settings

        self.bos_token_id = read_config["bos_token_id"]  # Note that the HF LlamaTokenizer doesn't seem to recognize these automatically
        self.eos_token_id = read_config["eos_token_id"]
        self.pad_token_id = read_config["pad_token_id"]

        self.hidden_size = read_config["hidden_size"]
        self.initializer_range = read_config["initializer_range"]
        self.intermediate_size = read_config["intermediate_size"]
        self.num_attention_heads = read_config["num_attention_heads"]
        self.num_hidden_layers = read_config["num_hidden_layers"]
        self.num_attention_heads = read_config["num_attention_heads"]
        self.rms_norm_eps = read_config["rms_norm_eps"]
        self.vocab_size = read_config["vocab_size"]

        self.rotary_embedding_base = 10000  # Constant used for pretrained models, leave as is unless retraining
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.groupsize = None  # Autodetected
        self.act_order = False  # Autodetected

        # Required settings

        self.model_path = None

        # Optional settings

        self.stream_layer_interval = 0  # Store every nth layer in system RAM and
        self.max_seq_len = 2048  # Reduce to save memory. Can also be increased, but the pretrained models produce degenerate output after 2048 tokens in any case. Should be possible to finetune for longer sequence lengths.
        self.attention_method = self.AttentionMethod.PYTORCH_SCALED_DP
        self.matmul_method = self.MatmulMethod.SWITCHED
        self.mlp_method = self.MLPMethod.NORMAL  # Currently no benefit to fused MLP
        self.device_map = ExLlamaDeviceMap(self.num_hidden_layers)
        self.auto_map = None  # List of ints with memory allocation in GB, per CUDA device, overrides device_map
        self.dequant = None  # Number of layers (per GPU) to de-quantize at load time


    # Parse and set list of GPU VRAM allocations

    def set_auto_map(self, map_string):

        if map_string is None: self.auto_map = None
        else: self.auto_map = [float(alloc) for alloc in map_string.split(",")]


    # Parse and set number of layers to de-quantize at load, per GPU

    def set_dequant(self, dq_string):

        if dq_string is None: self.dequant = None
        else: self.dequant = [int(alloc) for alloc in dq_string.split(",")]


def _dump_tensor(t, name):

    if t is None:
        with open(name, "w"):
            pass
        with open(name + ".shape", "w"):
            pass
    else:
        t.cpu().numpy().tofile(name)
        t = t.view(-1, t.shape[-1])
        with open(name + ".shape", "wb") as file:
            shape_struct = struct.pack("<ii", t.shape[0], t.shape[1])
            file.write(shape_struct)


# Switching

def _matmul_switch(config, x):

    if config.matmul_method == ExLlamaConfig.MatmulMethod.QUANT_ONLY: return False
    if config.matmul_method == ExLlamaConfig.MatmulMethod.PYTORCH_ONLY: return True

    xdp = 1
    for y in x.shape[:-1]: xdp *= y
    return xdp > optimal_switch_thd

def _mlp_switch(config, x):

    if config.act_order: return True  # Currently only implemented for no-act-order models
    if config.mlp_method == ExLlamaConfig.MLPMethod.FUSED: return False
    if config.mlp_method == ExLlamaConfig.MLPMethod.NORMAL: return True

    xdp = 1
    for y in x.shape[:-1]: xdp *= y
    return xdp > 1


# 4-bit linear layer implementation

class Ex4bitLinear(nn.Module):

    def __init__(self, config, in_features, out_features, has_bias, tensors, key, dequant = False):
        super().__init__()

        self.config = config
        self.key = key
        self.dequant = dequant

        self.in_features = in_features
        self.out_features = out_features
        self.bits = 4  # Only support 4 bits for now

        self.maxq = 2 ** self.bits - 1
        self.bias = None
        self.x_map = None
        self.seq_g_idx = None

        self.qweight = tensors[key + ".qweight"]

        self.qzeros = tensors[key + ".qzeros"]
        self.scales = tensors[key + ".scales"]

        # Infer groupsize from height of qzeros

        self.groupsize = None
        if self.qzeros.shape[0] > 1:

            self.groupsize = (self.qweight.shape[0] * 8) // self.qzeros.shape[0]

            if self.config.groupsize is None:
                self.config.groupsize = self.groupsize
            else:
                if self.config.groupsize != self.groupsize:
                    self.config.no_groupsize = True

        # Handle act-order matrix

        if key + ".g_idx" in tensors:

            if self.groupsize is None: raise ValueError("Found group index but no groupsize. What do?")

            self.config.act_order = True

            # Rearrange groups sequentially for act-order matrices

            g_idx = tensors[key + ".g_idx"]
            num_groups = self.qzeros.shape[0]
            seq_g_idx, self.x_map = cuda_ext.sequential_q4v2(self.qweight, g_idx, num_groups)

            # Discard group index if sequential groups all have the same groupsize. Treat as regular groupsize
            # matrix but keep the x_map

            i = 0
            j = 0
            discard = True
            while i < seq_g_idx.shape[-1]:
                if seq_g_idx[i].item() != j or seq_g_idx[i + 1].item() != self.groupsize:
                    discard = False
                    break
                i += self.groupsize * 2
                j += 1

            if not discard:

                self.seq_g_idx = seq_g_idx

        # Bias

        if has_bias: self.bias = tensors[key + ".bias"]

        # Optionally dequantize layer at init time

        if self.dequant:

            self.qweight_dequant = cuda_ext.dequantize_q4v2(self.quant_args())
            self.qweight = None
            self.scales = None
            self.zeros = None
            self.seq_g_idx = None
            self.x_map = None


    def quant_args(self):

        return {"qweight": self.qweight,
                "scales": self.scales,
                "zeros": self.qzeros,
                "seq_g_idx": self.seq_g_idx,
                "x_map": self.x_map}


    cpu_qweight: torch.Tensor
    cpu_scales: torch.Tensor
    cpu_qzeros: torch.Tensor
    cpu_seq_g_idx: torch.Tensor
    cpu_x_map: torch.Tensor

    def convert_streaming(self, stream_linear):

        # Copy tensors to CPU

        self.cpu_qweight = self.qweight.to("cpu")
        self.cpu_scales = self.scales.to("cpu")
        self.cpu_qzeros = self.qzeros.to("cpu")
        self.cpu_seq_g_idx = self.seq_g_idx.to("cpu") if self.seq_g_idx is not None else None
        self.x_map = self.x_map.to("cpu") if self.x_map is not None else None
        self.bias = self.bias.to("cpu") if self.x_map is not None else None

        # Replace reference with linear provided by stream buffer

        self.qweight = stream_linear.qweight
        self.scales = stream_linear.scales
        self.qzeros = stream_linear.qzeros
        self.seq_g_idx = stream_linear.seq_g_idx
        self.x_map = stream_linear.x_map
        self.bias = stream_linear.bias

        self.streaming = True


    streaming: bool = False
    is_loaded: bool = False

    def load_streaming(self):

        # Own references point to the same tensors as all other streamed linears, CPU copies are unique to this linear

        self.qweight.copy_(self.cpu_qweight, non_blocking = True)
        self.scales.copy_(self.cpu_scales, non_blocking = True)
        self.qzeros.copy_(self.cpu_qzeros, non_blocking = True)
        if self.seq_g_idx is not None: self.seq_g_idx.copy_(self.cpu_seq_g_idx, non_blocking = True)
        if self.x_map is not None: self.x_map.copy_(self.cpu_x_map, non_blocking = True)
        if self.bias is not None: self.bias.copy_(self.cpu_x_map, non_blocking = True)

        self.is_loaded = True


    def forward(self, x):

        if self.dequant:

            # out = torch.matmul(x, self.qweight_dequant)
            out = cuda_ext.matmul_half(x, self.qweight_dequant, _matmul_switch(self.config, x))
            # out = cuda_ext.matmul_half(x, self.qweight_dequant, True)

        else:

            if torch.is_grad_enabled():

                # Untested
                out = cuda_ext.ExAutogradMatmul4bitCuda.apply(x, self.qweight, self.scales, self.qzeros, self.groupsize, self.bits, self.maxq)

            else:

                out = cuda_ext.matmul_q4v2(x, self.quant_args(), _matmul_switch(self.config, x))
                if self.bias is not None: out += self.bias

            # if self.key == "model.layers.0.mlp.gate_proj":
            #
            #     _dump_tensor(x, "cuda_test/model.layers.0.mlp.gate_proj.x")
            #     sys.exit()

        return out


    def dump(self, filename):

        _dump_tensor(self.qweight, filename + ".qweight")
        _dump_tensor(self.scales, filename + ".scales")
        _dump_tensor(self.qzeros, filename + ".qzeros")
        _dump_tensor(self.seq_g_idx, filename + ".seq_g_idx")
        _dump_tensor(self.x_map, filename + ".x_map")
        _dump_tensor(self.bias, filename + ".bias")


# Llama MLP

class ExLlamaMLP(nn.Module):

    def __init__(self, config, tensors, key, dequant = False):
        super().__init__()

        self.config = config
        self.dequant = dequant

        self.gate_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.intermediate_size, False, tensors, key + ".gate_proj", dequant = dequant)
        self.up_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.intermediate_size, False, tensors, key + ".up_proj", dequant = dequant)
        self.down_proj = Ex4bitLinear(config, self.config.intermediate_size, self.config.hidden_size, False, tensors, key + ".down_proj", dequant = dequant)

        self.act_fn = nn.SiLU()


    def forward_fused(self, x, rms_norm_weight, buffer):

        assert not self.dequant
        x = cuda_ext.mlp_q4v2(x,
                              buffer.x_temp,
                              buffer.x_col_temp,
                              buffer.x_act_temp,
                              rms_norm_weight,
                              self.config.rms_norm_eps,
                              self.gate_proj.quant_args(),
                              self.up_proj.quant_args(),
                              self.down_proj.quant_args())

        return x


    def forward(self, x, buffer):

        y = self.gate_proj(x)
        y = self.act_fn(y)
        y *= self.up_proj(x)
        y = self.down_proj(y)

        return y

        # self.gate_proj.dump("cuda_test/mlp/gate_proj")
        # self.up_proj.dump("cuda_test/mlp/up_proj")
        # self.down_proj.dump("cuda_test/mlp/down_proj")
        # _dump_tensor(x, "cuda_test/mlp/test_mlp_x")
        # _dump_tensor(y, "cuda_test/mlp/test_mlp_x_gated")
        # _dump_tensor(x, "cuda_test/mlp/test_mlp_x_done")
        # sys.exit()


# RMS Layer norm.

class ExLlamaRMSNorm(nn.Module):

    def __init__(self, config, tensors, key):
        super().__init__()

        self.config = config
        self.variance_epsilon = self.config.rms_norm_eps
        self.weight = tensors[key]


    def forward(self, hidden_states, buffer):

        hidden_states = cuda_ext.llama_rms_norm(hidden_states, self.weight, self.variance_epsilon)
        return hidden_states


# Llama attention

class ExLlamaAttention(nn.Module):

    def __init__(self, config, tensors, key, sin, cos, index, dequant = False):
        super().__init__()

        self.config = config
        self.sin = sin
        self.cos = cos
        self.index = index

        self.q_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".q_proj", dequant = dequant)
        self.k_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".k_proj", dequant = dequant)
        self.v_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".v_proj", dequant = dequant)
        self.o_proj = Ex4bitLinear(config, self.config.num_attention_heads * self.config.head_dim, self.config.hidden_size, False, tensors, key + ".o_proj", dequant = dequant)


    def forward(self, hidden_states, cache, buffer):

        bsz, q_len, _ = hidden_states.size()
        past_len = cache.current_seq_len

        # Project q, k, v

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)

        # Apply position embeddings

        cos_emb = self.cos.narrow(2, past_len, q_len)
        sin_emb = self.sin.narrow(2, past_len, q_len)

        def rotate_half(x):
            half_size = x.shape[-1] // 2
            x1 = x.narrow(-1, 0, half_size)
            x2 = x.narrow(-1, half_size, half_size)
            return torch.cat((-x2, x1), dim = -1)

        query_states_r = rotate_half(query_states)
        query_states_r.mul_(sin_emb)
        query_states.mul_(cos_emb)
        query_states.add_(query_states_r)

        key_states_r = rotate_half(key_states)
        key_states_r.mul_(sin_emb)
        key_states.mul_(cos_emb)
        key_states.add_(key_states_r)

        # Add keys and values to cache

        new_keys = cache.key_states[self.index].narrow(2, past_len, q_len)
        new_values = cache.value_states[self.index].narrow(2, past_len, q_len)
        new_keys.copy_(key_states)
        new_values.copy_(value_states)

        # Key/value tensors with past

        key_states = cache.key_states[self.index].narrow(2, 0, past_len + q_len)
        value_states = cache.value_states[self.index].narrow(2, 0, past_len + q_len)

        # Attention

        # -- HF Transformers regular attention, O(n^2) memory usage, bunch of mallocs

        if self.config.attention_method == ExLlamaConfig.AttentionMethod.PYTORCH_MATMUL:

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.config.head_dim)
            attn_weights = attn_weights + buffer.attn_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2)
            del attn_weights

        # -- Scaled dot-product attention from PyTorch 2, should be comparable to xformers (?)

        elif self.config.attention_method == ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP:

            # Torch's SDP attention has a built-in causal mask feature which we can use only when there is no past, i.e.
            # it can only apply a square attention mask. It saves quite a bit of VRAM but in practice Torch seems to use
            # the same amount of memory at peak anyway. It's also a little slower, and it gives misleading benchmarks
            # since it doesn't actually apply in the case we're interested in (autoregression.) Disabled for now.

            if True or past_len > 0:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = buffer.attn_mask, is_causal = False)
            else:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = None, is_causal = True)

            attn_output = attn_output.transpose(1, 2)

        else: raise ValueError("Wut?")

        # Output projection

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class ExLlamaDecoderLayer(nn.Module):

    def __init__(self, config, tensors, key, index, sin, cos, dequant = False):
        super().__init__()

        self.config = config
        self.index = index

        self.self_attn = ExLlamaAttention(self.config, tensors, key + ".self_attn", sin, cos, self.index, dequant = dequant)
        self.mlp = ExLlamaMLP(self.config, tensors, key + ".mlp", dequant = dequant)

        self.input_layernorm = ExLlamaRMSNorm(self.config, tensors, key + ".input_layernorm.weight")
        self.post_attention_layernorm = ExLlamaRMSNorm(self.config, tensors, key + ".post_attention_layernorm.weight")


    def forward(self, hidden_states, cache, buffer):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, buffer)
        hidden_states = self.self_attn(hidden_states, cache, buffer)
        hidden_states = residual + hidden_states

        # TODO: Support dequantized layer in fused MLP. Also, finish implementing fused MLP

        if self.mlp.dequant or _mlp_switch(self.config, hidden_states):

            residual = hidden_states
            hidden_states = self.post_attention_layernorm(hidden_states, buffer)
            hidden_states = self.mlp(hidden_states, buffer)
            hidden_states = residual + hidden_states

        else:

            hidden_states = self.mlp.forward_fused(hidden_states, self.post_attention_layernorm.weight, buffer)

        return hidden_states

        # _dump_tensor(hidden_states, "cuda_test/mlp/test_mlp_x_prenorm")
        # _dump_tensor(self.post_attention_layernorm.weight, "cuda_test/mlp/test_mlp_norm_weight")
        # _dump_tensor(hidden_states, "cuda_test/mlp/test_mlp_x_postresidual")


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

                p_key_states = torch.zeros(self.batch_size, self.config.num_attention_heads, self.max_seq_len, self.config.head_dim, dtype = torch.float16, device = self.model.config.device_map.layers[i])
                p_value_states = torch.zeros(self.batch_size, self.config.num_attention_heads, self.max_seq_len, self.config.head_dim, dtype = torch.float16, device = self.model.config.device_map.layers[i])

            else:

                p_key_states = copy_from.key_states[i].clone()
                p_value_states = copy_from.value_states[i].clone()

            self.key_states.append(p_key_states)
            self.value_states.append(p_value_states)


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


    def debug(self):

        print(self.current_seq_len, self.key_states[0][0, 0, :self.current_seq_len, :])


# Layer streaming
# TODO: Currently assumes single GPU

class ExLlamaStreamer:

    mlp_gate_proj: Ex4bitLinear
    mlp_up_proj: Ex4bitLinear
    mlp_down_proj: Ex4bitLinear

    self_attn_q_proj: Ex4bitLinear
    self_attn_k_proj: Ex4bitLinear
    self_attn_v_proj: Ex4bitLinear
    self_attn_o_proj: Ex4bitLinear

    # Copy the first layer to be streamed

    def __init__(self, config, layer):

        self.config = config

        # Reference all relevant tensors in the first layer. All linears in subsequent layers will reference these
        # and dereference their own while maintaining CPU copies

        self.mlp_gate_proj = layer.mlp.gate_proj
        self.mlp_up_proj = layer.mlp.up_proj
        self.mlp_down_proj = layer.mlp.down_proj

        self.self_attn_q_proj = layer.self_attn.q_proj
        self.self_attn_k_proj = layer.self_attn.k_proj
        self.self_attn_v_proj = layer.self_attn.v_proj
        self.self_attn_o_proj = layer.self_attn.o_proj

        # Separate CUDA stream for background transfer
        # TODO: Just using first stream layer device, we really need a stream buffer per device

        self.cuda_stream = torch.cuda.Stream(self.mlp_gate_proj.qweight.device)


    # Set up layer for streaming

    def convert_linear(self, self_linear, linear):

        assert self_linear.qweight.shape == linear.qweight.shape
        assert self_linear.scales.shape == linear.scales.shape
        assert self_linear.qzeros.shape == linear.qzeros.shape
        assert self_linear.seq_g_idx is None or self_linear.seq_g_idx.shape == linear.seq_g_idx.shape
        assert self_linear.x_map is None or self_linear.x_map.shape == linear.x_map.shape
        assert self_linear.bias is None or self_linear.bias.shape == linear.bias.shape

        linear.convert_streaming(self_linear)


    def convert_layer(self, layer):

        self.convert_linear(self.mlp_gate_proj, layer.mlp.gate_proj)
        self.convert_linear(self.mlp_up_proj, layer.mlp.up_proj)
        self.convert_linear(self.mlp_down_proj, layer.mlp.down_proj)

        self.convert_linear(self.self_attn_q_proj, layer.self_attn.q_proj)
        self.convert_linear(self.self_attn_k_proj, layer.self_attn.k_proj)
        self.convert_linear(self.self_attn_v_proj, layer.self_attn.v_proj)
        self.convert_linear(self.self_attn_o_proj, layer.self_attn.o_proj)


    # Load layer

    def load_layer(self, layer):

        with torch.cuda.stream(self.cuda_stream):

            layer.mlp.gate_proj.load_streaming()
            layer.mlp.up_proj.load_streaming()
            layer.mlp.down_proj.load_streaming()

            layer.self_attn.q_proj.load_streaming()
            layer.self_attn.k_proj.load_streaming()
            layer.self_attn.v_proj.load_streaming()
            layer.self_attn.o_proj.load_streaming()


    def load_layer_sync(self):

        self.cuda_stream.synchronize()


# Device map for the model.

class ExLlamaDeviceMap:

    def __init__(self, num_layers):

        self.num_layers = num_layers

        self.embed_tokens = "cpu"  # Embedding table on CPU saves 400 MB on the 30B model with no measurable impact on performance
        self.lm_head = "cuda:0"
        self.norm = "cuda:0"
        self.layers = ["cuda:0"] * self.num_layers
        self.stream_layer_interval = 0


    def get_layers_devs(self):

        return list(set(self.layers))


    def map(self, key, loading = False):

        if key.startswith("lm_head."): return self.lm_head
        if key.startswith("model.embed_tokens."): return self.embed_tokens
        if key.startswith("model.norm."): return self.norm

        if key.startswith("model.layers."):
            num = int(key.split(".")[2])
            if loading and self.stream_layer_interval > 0 and (num + 1) % self.stream_layer_interval == 0:
                if key.startswith(f"model.layers.{num}.mlp."): return "cpu"
                if key.startswith(f"model.layers.{num}.self_attn."): return "cpu"
            return self.layers[num]

        raise ValueError("Unknown key: " + key)


class ExLlamaBuffer:

    config: ExLlamaConfig

    def __init__(self, config):

        self.config = config

    # Attention mask

    attn_mask: torch.Tensor = None

    # Fused MLP

    x_temp: torch.Tensor = None
    x_col_temp: torch.Tensor = None
    x_act_temp: torch.Tensor = None

    def prepare_fused_mlp(self, hidden_state, first_device):

        self.x_temp = torch.empty(hidden_state.shape, device = first_device, dtype = torch.float16)
        # self.x_temp = torch.empty((1, self.config.max_seq_len, self.config.hidden_size), device = first_device, dtype = torch.float16)
        self.x_col_temp = torch.empty((self.x_temp.shape[0],), device = first_device, dtype = torch.float32)
        self.x_act_temp = torch.empty(self.x_temp.shape[:-1] + (self.config.intermediate_size,), device = first_device, dtype = torch.float16)

    # Move to device

    def to(self, device):

        new = ExLlamaBuffer(self.config)

        new.x_temp = None if self.x_temp is None else self.x_temp.to(device)
        new.x_col_temp = None if self.x_col_temp is None else self.x_temp.to(device)
        new.x_act_temp = None if self.x_act_temp is None else self.x_temp.to(device)

        return new


def _device_to_int(device):

    return int(device[device.find(":") + 1:])

def _skip_key(key):

    if key.endswith("_proj.bias"): return True
    if key.endswith(".rotary_emb.inv_freq"): return True
    return False


class ExLlama(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.eval()

        self.config = config
        self.stream_buffer = None

        # Forward streaming config to device map so we only load the first layer on GPU

        self.config.device_map.stream_layer_interval = self.config.stream_layer_interval

        # Load model weights

        tensors = {}
        with safe_open(self.config.model_path, framework="pt", device="cpu") as f:

            # Begin auto mapping if enabled

            decoder_size = 0
            decoder_dq_size = 0
            norm_size = 0
            head_size = 0
            half_element_size = torch.tensor([], dtype = torch.float16).element_size()

            if self.config.auto_map is not None:

                self.config.device_map.embed_tokens = "cpu"
                self.config.device_map.layers = ["cuda:0"] + ["?"] * (self.config.num_hidden_layers - 1)

                for key in f.keys():

                    if _skip_key(key): continue

                    if key.startswith("model.layers.0."):
                        tensor = f.get_tensor(key)
                        decoder_size += tensor.numel() * tensor.element_size()
                        if key.endswith(".weight"):
                            decoder_dq_size += tensor.numel() * tensor.element_size()
                        if key.endswith(".qweight"):
                            decoder_dq_size += tensor.numel() * 8 * half_element_size

                    if key.startswith("model.norm."):
                        tensor = f.get_tensor(key)
                        norm_size += tensor.numel() * tensor.element_size()

                    if key.startswith("lm_head."):
                        tensor = f.get_tensor(key)
                        head_size += tensor.numel() * tensor.element_size()

                # Assign layers automatically

                device_usage = 0
                device_index = 0
                layer_index_device = 0
                max_usage = self.config.auto_map[device_index] * (1024 ** 3)

                for layer in range(self.config.num_hidden_layers + 2):

                    this_layer_size = decoder_size
                    if layer == self.config.num_hidden_layers + 0: this_layer_size = norm_size
                    elif layer == self.config.num_hidden_layers + 1: this_layer_size = head_size
                    elif self.config.dequant is not None and layer_index_device < self.config.dequant[device_index]: this_layer_size = decoder_dq_size

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

            # Load tensors, move to device(s)

            for key in f.keys():

                if _skip_key(key): continue

                device = self.config.device_map.map(key, loading = True)
                tensor = f.get_tensor(key)

                if key.endswith(".scales"): tensor = tensor.half()
                if key == "lm_head.weight": tensor = tensor.float() if device == "cpu" else tensor.half()
                if key == "model.norm.weight": tensor = tensor.half()
                if key.endswith(".embed_tokens.weight"): tensor = tensor.half()
                if key.endswith(".input_layernorm.weight"): tensor = tensor.half()
                if key.endswith(".post_attention_layernorm.weight"): tensor = tensor.half()

                tensor = tensor.to(device, non_blocking = True)
                tensors[key] = tensor
                # print(key + " -> " + device)

        # Head

        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False, device = "meta")
        self.lm_head.weight = nn.Parameter(tensors["lm_head.weight"])
        # self.lm_head_data = tensors["lm_head.weight"].transpose(0, 1).contiguous()

        # Token embeddings

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id, device = "meta")
        self.embed_tokens.weight = nn.Parameter(tensors["model.embed_tokens.weight"])

        # Norm

        self.norm = ExLlamaRMSNorm(self.config, tensors, "model.norm.weight")

        # Prepare position embeddings for max seq length

        devs = self.config.device_map.get_layers_devs()

        self.sincos = {}
        for device in devs:

            inv_freq = 1.0 / (self.config.rotary_embedding_base ** (torch.arange(0, self.config.head_dim, 2, device = device).float() / self.config.head_dim))
            t = torch.arange(self.config.max_seq_len, device = device, dtype = torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)

            sin = emb.sin()[None, None, :, :].half()
            cos = emb.cos()[None, None, :, :].half()

            self.sincos[device] = (sin, cos)

        # Layers

        layer_streaming = self.config.stream_layer_interval > 0

        modules = []
        device_layer_index = [0] * len(devs)

        for i in range(self.config.num_hidden_layers):

            device = self.config.device_map.layers[i]
            sin, cos = self.sincos[device]

            dequant = False
            if self.config.dequant is not None:
                device_idx = _device_to_int(device)
                device_layer = device_layer_index[device_idx]
                device_layer_index[device_idx] += 1
                if device_layer < self.config.dequant[device_idx]: dequant = True

            layer = ExLlamaDecoderLayer(self.config, tensors, f"model.layers.{i}", i, sin, cos, dequant = dequant)

            if layer_streaming and i > 0 and (i + 1) % self.config.stream_layer_interval == 0:
                if self.stream_buffer is None: self.stream_buffer = ExLlamaStreamer(self.config, layer)  # Use first layer as prototype
                self.stream_buffer.convert_layer(layer)

            modules.append(layer)

        self.layers = nn.ModuleList(modules)


    def forward(self, input_ids, cache, last_id_only = True, preprocess_only = False):

        batch_size, seq_len = input_ids.shape
        past_len = cache.current_seq_len

        buffer = ExLlamaBuffer(self.config)

        # Build attention mask on first device, copy to others if necessary

        devs = self.config.device_map.get_layers_devs()

        attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len + past_len, dtype = torch.float16, device = devs[0])

        if seq_len > 1:
            attn_mask = torch.zeros(batch_size, 1, seq_len, past_len + seq_len, dtype = torch.float16, device = devs[0])
            attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), torch.finfo(torch.float16).min))
            attn_mask[:, :, : seq_len - 1, past_len + 1: past_len + seq_len] = attn_mask_triu

        buffer.attn_mask = attn_mask

        # Embeddings
        # TODO: Allow passing input embeddings instead of IDs

        hidden_states = self.embed_tokens(input_ids.to(self.config.device_map.embed_tokens))

        # Prepare fused MLP buffers if not switching

        if not _mlp_switch(self.config, hidden_states):

            buffer.prepare_fused_mlp(hidden_states, devs[0])

        # Split buffers to devices

        buffers = {devs[0]: buffer}
        for device in devs[1:]: buffers[device] = buffer.to(device)

        # Decoder layers

        next_streaming_layer = -1
        background_thread = None
        layer_streaming = self.config.stream_layer_interval > 0

        if layer_streaming:

            next_streaming_layer = self.config.stream_layer_interval - 1
            # background_thread = threading.Thread(target = self.stream_buffer.load_layer, args = (self.layers[next_streaming_layer],))
            # background_thread.start()
            self.stream_buffer.load_layer(self.layers[next_streaming_layer])


        for i, decoder_layer in enumerate(self.layers):

            device = self.config.device_map.layers[i]
            hidden_states = hidden_states.to(device)

            if i == next_streaming_layer:

                # background_thread.join()
                self.stream_buffer.load_layer_sync()

                hidden_states = decoder_layer(hidden_states, cache, buffers[device])

                next_streaming_layer += self.config.stream_layer_interval
                if next_streaming_layer < len(self.layers):
                    # background_thread = threading.Thread(target = self.stream_buffer.load_layer, args = (self.layers[next_streaming_layer],))
                    # background_thread.start()
                    torch.cuda.synchronize()  # Need to let last streamed layer finish with the buffers
                    self.stream_buffer.load_layer(self.layers[next_streaming_layer])

            else:

                hidden_states = decoder_layer(hidden_states, cache, buffers[device])

        cache.current_seq_len += seq_len

        # Early exit when we don't need logits

        if preprocess_only: return None

        # Norm

        hidden_states = hidden_states.to(self.config.device_map.norm)
        hidden_states = self.norm(hidden_states, buffer)

        # Head

        if last_id_only: hidden_states = hidden_states[:, -1:, :]

        hidden_states = hidden_states.to(self.config.device_map.lm_head)
        if self.config.device_map.lm_head == "cpu": hidden_states = hidden_states.float()
        logits = self.lm_head(hidden_states)
        # logits = cuda_ext.matmul_half(hidden_states, self.lm_head_data, cublas = False)

        return logits.float().to(self.config.device_map.embed_tokens)

        # TODO: Accept labels and calc (optional) loss, also test backprop
        # HF implementation for ref.:
        #
        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)
