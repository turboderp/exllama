import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
import cuda_ext
import json
import math
from enum import Enum
import struct

# Magic numbers

optimal_switch_thd = 6  # Model mostly runs one token at a time, or many. So this probably doesn't matter too much.
attn_switch_thd = 16

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
        self.device_map = ExLlamaDeviceMap(self.num_hidden_layers)

        # Optional settings

        self.max_seq_len = 2048  # Reduce to save memory. Can also be increased, but the pretrained models produce degenerate output after 2048 tokens in any case. Should be possible to finetune for longer sequence lengths.
        self.gpu_peer_fix = False # Apparently Torch can have problems transferring tensors directly one GPU to another sometimes. Enable this to move tensors via system RAM instead, where needed
        self.auto_map = None  # List of ints with memory allocation in GB, per CUDA device, overrides device_map
        self.debug = False

        # Tuning

        self.matmul_recons_thd = 8
        self.fused_mlp_thd = 2
        self.sdp_thd = 8
        self.matmul_fused_remap = False
        self.rmsnorm_no_half2 = False
        self.rope_no_half2 = False
        self.matmul_no_half2 = False
        self.silu_no_half2 = False

    # Copy tuning params to C++ extension

    def set_tuning_params(self):

        cuda_ext.exllama_ext.set_tuning_params(self.matmul_recons_thd,
                                               self.fused_mlp_thd,
                                               self.sdp_thd,
                                               self.matmul_fused_remap,
                                               self.rmsnorm_no_half2,
                                               self.rope_no_half2,
                                               self.matmul_no_half2,
                                               self.silu_no_half2)

    # Parse and set list of GPU VRAM allocations

    def set_auto_map(self, map_string):

        if map_string is None: self.auto_map = None
        else: self.auto_map = [float(alloc) for alloc in map_string.split(",")]


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


# 4-bit linear layer implementation

class Ex4bitLinear:
# class Ex4bitLinear(nn.Module):

    def __init__(self, config, in_features, out_features, has_bias, tensors, key):
        # super().__init__()

        self.config = config
        self.key = key
        self.in_features = in_features
        self.out_features = out_features

        self.qweight = tensors[key + ".qweight"]
        self.qzeros = tensors[key + ".qzeros"]
        self.scales = tensors[key + ".scales"]
        self.g_idx = tensors[key + ".g_idx"].cpu() if key + ".g_idx" in tensors else None
        self.bias = tensors[key + ".bias"] if has_bias else None

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


    def forward(self, x):

        out = cuda_ext.ext_q4_matmul(x, self.q4, self.width)
        if self.bias is not None: out.add_(self.bias)
        return out


    def debug(self, name):

        print(f" !!  - {name}: {self.qweight.device} ", end = "")
        print(f"[Q", end = "")
        if self.x_map is not None: print(f",x_map", end = "")
        if self.bias is not None: print(f",bias", end = "")
        print(f"]", end = "")
        unique_q = torch.unique(self.qweight[0, :]).numel()
        width_q = self.qweight.shape[-1]
        if unique_q < width_q / 2: print(f" ### qweight abnormal", end = "")
        unique_z = torch.unique(self.qzeros[0, :]).numel()
        width_z = self.qzeros.shape[-1]
        if unique_z < width_z / 8: print(f" ### qzeros abnormal", end = "")
        print(f" scales min/max/std: {self.scales.min().item():.6f}/{self.scales.max().item():.6f}/{self.scales.std().item():.6f}", end = "")
        print("")


# Llama MLP

class ExLlamaMLP:
# class ExLlamaMLP(nn.Module):

    def __init__(self, config, tensors, key):
        # super().__init__()

        self.config = config

        self.gate_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.intermediate_size, False, tensors, key + ".gate_proj")
        self.up_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.intermediate_size, False, tensors, key + ".up_proj")
        self.down_proj = Ex4bitLinear(config, self.config.intermediate_size, self.config.hidden_size, False, tensors, key + ".down_proj")

        self.act_fn = nn.SiLU()


    def forward(self, x, buffer):

        y = self.gate_proj.forward(x)
        y = self.act_fn(y)
        y *= self.up_proj.forward(x)
        y = self.down_proj.forward(y)

        return y


# RMS Layer norm.

class ExLlamaRMSNorm:
# class ExLlamaRMSNorm(nn.Module):

    def __init__(self, config, tensors, key):
        # super().__init__()

        self.config = config
        self.variance_epsilon = self.config.rms_norm_eps
        self.weight = tensors[key]


    def forward(self, hidden_states, buffer):

        hidden_states = cuda_ext.ext_rms_norm(hidden_states, self.weight, self.variance_epsilon)
        return hidden_states


    def debug(self):

        print(f" !!  - layernorm.weight: {_describe_tensor(self.weight, True)} eps: {self.variance_epsilon:.8f}")


# Llama attention

class ExLlamaAttention:
# class ExLlamaAttention(nn.Module):

    def __init__(self, config, tensors, key, sin, cos, index):
        # super().__init__()

        self.config = config
        self.sin = sin
        self.cos = cos
        self.index = index

        self.q_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".q_proj")
        self.k_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".k_proj")
        self.v_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".v_proj")
        self.o_proj = Ex4bitLinear(config, self.config.num_attention_heads * self.config.head_dim, self.config.hidden_size, False, tensors, key + ".o_proj")


    def forward(self, hidden_states, cache, buffer):

        bsz, q_len, _ = hidden_states.size()
        past_len = cache.current_seq_len

        # Project q, k, v, apply position embeddings to k and v

        query_states = self.q_proj.forward(hidden_states)
        key_states = self.k_proj.forward(hidden_states)

        cuda_ext.exllama_ext.rope_(query_states, self.sin, self.cos, past_len, self.config.num_attention_heads, self.config.head_dim)
        cuda_ext.exllama_ext.rope_(key_states, self.sin, self.cos, past_len, self.config.num_attention_heads, self.config.head_dim)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)
        value_states = self.v_proj.forward(hidden_states).view(bsz, q_len, self.config.num_attention_heads, self.config.head_dim).transpose(1, 2)

        # Add keys and values to cache

        if self.config.debug: cache.debug(self.index)

        new_keys = cache.key_states[self.index].narrow(2, past_len, q_len)
        new_values = cache.value_states[self.index].narrow(2, past_len, q_len)
        new_keys.copy_(key_states)
        new_values.copy_(value_states)

        # Key/value tensors with past

        key_states = cache.key_states[self.index].narrow(2, 0, past_len + q_len)
        value_states = cache.value_states[self.index].narrow(2, 0, past_len + q_len)

        # Attention

        # -- HF Transformers regular attention, faster on shorter sequences, same VRAM usage

        if self.config.sdp_thd == 0 or q_len < self.config.sdp_thd:

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
            attn_weights /= math.sqrt(self.config.head_dim)
            if buffer.attn_mask is not None and buffer.attn_mask.shape[2] > 1: attn_weights = attn_weights + buffer.attn_mask
            # attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2)

        # -- Scaled dot-product attention from PyTorch 2, should be comparable to xformers (?)

        else:

            # Torch's SDP attention has a built-in causal mask feature which we can use only when there is no past, i.e.
            # it can only apply a square attention mask. It saves quite a bit of VRAM but in practice Torch seems to use
            # the same amount of memory at peak anyway.

            if past_len > 0:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = buffer.attn_mask, is_causal = False)
            else:
                attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = None, is_causal = True)

            attn_output = attn_output.transpose(1, 2)

        # Output projection

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj.forward(attn_output)

        return attn_output


def _rows(x):
    xdp = 1
    for y in x.shape[:-1]: xdp *= y
    return xdp

class ExLlamaDecoderLayer:
# class ExLlamaDecoderLayer(nn.Module):

    def __init__(self, config, tensors, key, index, sin, cos):
        # super().__init__()

        self.config = config
        self.index = index

        self.self_attn = ExLlamaAttention(self.config, tensors, key + ".self_attn", sin, cos, self.index)
        self.mlp = ExLlamaMLP(self.config, tensors, key + ".mlp")

        self.input_layernorm = ExLlamaRMSNorm(self.config, tensors, key + ".input_layernorm.weight")
        self.post_attention_layernorm = ExLlamaRMSNorm(self.config, tensors, key + ".post_attention_layernorm.weight")


    def forward(self, hidden_states, cache, buffer):

        if self.config.debug:
            print(f" !! Begin decoder {self.index}")
            print(f" !! Begin self-attention")
            print(f" !!  - hidden_states: {_describe_tensor(hidden_states, True)}")
            buffer.debug()
            self.input_layernorm.debug()
            self.self_attn.q_proj.debug("self_attn.q_proj")
            self.self_attn.k_proj.debug("self_attn.k_proj")
            self.self_attn.v_proj.debug("self_attn.v_proj")
            self.self_attn.o_proj.debug("self_attn.o_proj")

        residual = hidden_states
        hidden_states = self.input_layernorm.forward(hidden_states, buffer)
        hidden_states = self.self_attn.forward(hidden_states, cache, buffer)
        hidden_states = residual + hidden_states

        if self.config.debug:

            print(f" !! Begin MLP")
            print(f" !!  - hidden_states: {_describe_tensor(hidden_states, True)}")
            self.post_attention_layernorm.debug()
            self.mlp.gate_proj.debug("mlp.gate_proj")
            self.mlp.up_proj.debug("mlp.up_proj")
            self.mlp.down_proj.debug("mlp.down_proj")

        if self.config.fused_mlp_thd > 0 and _rows(hidden_states) <= self.config.fused_mlp_thd:

            if self.config.debug: print(f" !!  - method: fused")

            cuda_ext.ext_q4_mlp(hidden_states,
                                self.post_attention_layernorm.weight,
                                self.config.rms_norm_eps,
                                self.mlp.gate_proj.q4,
                                self.mlp.up_proj.q4,
                                self.mlp.down_proj.q4)

        else:

            if self.config.debug: print(f" !!  - method: normal")

            residual = hidden_states
            hidden_states = self.post_attention_layernorm.forward(hidden_states, buffer)
            hidden_states = self.mlp.forward(hidden_states, buffer)
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


    def debug(self, index):

        print(f" !!  - cache device: {self.key_states[index].device}, seq_len: {self.current_seq_len}")


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


    def map(self, key, loading = False):

        if key.startswith("lm_head."): return self.lm_head
        if key.startswith("model.embed_tokens."): return self.embed_tokens
        if key.startswith("model.norm."): return self.norm

        if key.startswith("model.layers."):
            num = int(key.split(".")[2])
            return self.layers[num]

        raise ValueError("Unknown key: " + key)


    def debug(self):

        print(f" !! Device map:")
        print(f" !!  - embed_tokens: {self.embed_tokens}")

        a = 0
        while a < len(self.layers):
            b = min(a + 10, len(self.layers))
            print(f" !!  - layers [{a}:{b}]:", end = "")
            for i in range(a, b): print(" " + self.layers[i], end = "")
            print("")
            a = b

        print(f" !!  - norm: {self.norm}")
        print(f" !!  - lm_head: {self.lm_head}")


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


    def debug(self):

        print(f" !!  - attn_mask: {_describe_tensor(self.attn_mask, True)}")


def _device_to_int(device):

    return int(device[device.find(":") + 1:])

def _skip_key(key):

    if key.endswith("_proj.bias"): return True
    if key.endswith(".rotary_emb.inv_freq"): return True
    return False

def _describe_tensor(tensor, stats = False):

    if tensor is None: return "None"
    desc = f"device: {tensor.device}"
    desc += f", shape: {str(list(tensor.shape))}"
    desc += f", dtype: {str(tensor.dtype).replace('torch.', '')}"
    if stats:
        if tensor.dtype in (torch.float16, torch.float32, torch.float64):
            desc += f", min: {tensor.min().item():.6f}"
            desc += f", max: {tensor.max().item():.6f}"
            desc += f", std: {tensor.std().item():.6f}"
        else:
            desc += f", min: {tensor.min().item()}"
            desc += f", max: {tensor.max().item()}"
    return desc

def _move_tensor(tensor, new_device, name, config):
    device = str(tensor.device)
    if device == new_device: return tensor
    if config.debug: print(f" !! Moving {name} from {device} to {new_device}")
    if config.gpu_peer_fix:
        if device.startswith("cuda:") and new_device.startswith("cuda:"):
            tensor = tensor.to("cpu")
    return tensor.to(new_device)


class ExLlama:
# class ExLlama(nn.Module):

    def __init__(self, config):
        # super().__init__()
        # self.eval()

        self.config = config

        if self.config.debug:
            device_count = torch.cuda.device_count()
            print(f" !! Available CUDA devices:")
            for i in range(device_count):
                print(f'" !!  - cuda:{i}: {torch.cuda.get_device_name(i)}')

        # Copy tuning parameters to C++ extension

        self.config.set_tuning_params()

        # Load model weights

        if self.config.debug: print(f" !! Loading safetensors file: {self.config.model_path}")

        tensors = {}
        with safe_open(self.config.model_path, framework="pt", device="cpu") as f:

            # Begin auto mapping if enabled

            decoder_size = 0
            norm_size = 0
            head_size = 0
            half_element_size = torch.tensor([], dtype = torch.float16).element_size()

            if self.config.auto_map is not None:

                if self.config.debug: print(f" !! Begin auto device map")

                self.config.device_map.embed_tokens = "cpu"
                self.config.device_map.layers = ["cuda:0"] + ["?"] * (self.config.num_hidden_layers - 1)

                for key in f.keys():

                    if _skip_key(key): continue

                    if key.startswith("model.layers.0."):
                        tensor = f.get_tensor(key)
                        decoder_size += tensor.numel() * tensor.element_size()

                    if key.startswith("model.norm."):
                        tensor = f.get_tensor(key)
                        norm_size += tensor.numel() * tensor.element_size()

                    if key.startswith("lm_head."):
                        tensor = f.get_tensor(key)
                        head_size += tensor.numel() * tensor.element_size()

                if self.config.debug:
                    print(f" !! Decoder size: {decoder_size:,} bytes")
                    print(f" !! Norm size: {norm_size:,} bytes")
                    print(f" !! Head size: {head_size:,} bytes")

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

                if self.config.debug: self.config.device_map.debug()

            # Load tensors, move to device(s)

            max_dq_buffer_size = 0

            if self.config.debug: print(f" !! Begin load tensors")
            for key in f.keys():

                if _skip_key(key): continue

                device = self.config.device_map.map(key, loading = True)
                tensor = f.get_tensor(key)

                if self.config.debug:
                    if key.startswith("model.layers.0.") or not key.startswith("model.layers."):
                        print(f" !!  - {key} read: {_describe_tensor(tensor)}")

                if key.endswith(".scales"): tensor = tensor.half()
                if key == "lm_head.weight": tensor = tensor.float() if device == "cpu" else tensor.half()
                if key == "model.norm.weight": tensor = tensor.half()
                if key.endswith(".embed_tokens.weight"): tensor = tensor.half()
                if key.endswith(".input_layernorm.weight"): tensor = tensor.half()
                if key.endswith(".post_attention_layernorm.weight"): tensor = tensor.half()

                tensor = tensor.to(device, non_blocking = True)

                if key.endswith(".qweight"): max_dq_buffer_size = max(max_dq_buffer_size, tensor.numel() * 8)

                if self.config.debug:
                    if key.startswith("model.layers.0.") or not key.startswith("model.layers."):
                        print(f" !!  - {key} map: {_describe_tensor(tensor, device.startswith('cuda'))}")

                tensors[key] = tensor

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

        if self.config.debug: print(f" !! Computing RoPE table for seq length: {self.config.max_seq_len}")

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
            if self.config.debug: print(f" !! - stored for device: {device}")

        # Layers

        modules = []
        device_layer_index = [0] * len(devs)

        for i in range(self.config.num_hidden_layers):

            device = self.config.device_map.layers[i]
            sin, cos = self.sincos[device]

            layer = ExLlamaDecoderLayer(self.config, tensors, f"model.layers.{i}", i, sin, cos)

            modules.append(layer)

        self.layers = modules
        # self.layers = nn.ModuleList(modules)

        # Prepare CUDA buffers

        self.buffers = []
        for dev in self.config.device_map.get_layers_devs():

            device_buffers = {}
            self.buffers.append(device_buffers)

            temp_state = torch.zeros((config.max_seq_len, config.intermediate_size), dtype = torch.float16, device = dev)
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


    def forward(self, input_ids, cache, last_id_only = True, preprocess_only = False):

        if torch.is_grad_enabled():
            raise ValueError("Forward pass called with gradients enabled. Back propagation is not supported yet.")

        if self.config.debug: print(f" !! Begin forward pass")

        batch_size, seq_len = input_ids.shape
        past_len = cache.current_seq_len

        buffer = ExLlamaBuffer(self.config)

        # Build attention mask on first device, copy to others if necessary

        devs = self.config.device_map.get_layers_devs()

        if seq_len > 1:

            attn_mask = torch.zeros(batch_size, 1, seq_len, past_len + seq_len, dtype = torch.float16, device = devs[0])
            attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), torch.finfo(torch.float16).min))
            attn_mask[:, :, : seq_len - 1, past_len + 1: past_len + seq_len] = attn_mask_triu

        else:

            attn_mask = None
            # attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len + past_len, dtype = torch.float16, device = devs[0])

        buffer.attn_mask = attn_mask

        # Embeddings
        # TODO: Allow passing input embeddings instead of IDs

        input_ids = _move_tensor(input_ids, "cpu", "input_ids", self.config)
        hidden_states = self.embed_tokens(input_ids)

        if self.config.debug: print(f" !! Built initial hidden state: {_describe_tensor(hidden_states, True)}")

        # Split buffers to devices

        buffers = {devs[0]: buffer}
        for device in devs[1:]:
            buffers[device] = buffer.to(device)

        if self.config.debug:
            for device in devs:
                print(f" !! Prepared buffer for device: {device}")
                buffers[device].debug()

        # Decoder layers

        for i, decoder_layer in enumerate(self.layers):

            device = self.config.device_map.layers[i]
            hidden_states = _move_tensor(hidden_states, device, "hidden_states", self.config)

            hidden_states = decoder_layer.forward(hidden_states, cache, buffers[device])

        cache.current_seq_len += seq_len

        # Early exit when we don't need logits

        if preprocess_only: return None

        # Norm

        hidden_states = _move_tensor(hidden_states, self.config.device_map.norm, "hidden_states", self.config)

        if self.config.debug: print(f" !! pre norm, hidden_states: {_describe_tensor(hidden_states, True)}")

        hidden_states = self.norm.forward(hidden_states, buffer)

        # Head

        if last_id_only: hidden_states = hidden_states[:, -1:, :].contiguous()
        if self.config.device_map.lm_head == "cpu": hidden_states = hidden_states.float()

        hidden_states = _move_tensor(hidden_states, self.config.device_map.lm_head, "hidden_states", self.config)

        if self.config.debug: print(f" !! pre lm_head, hidden_states: {_describe_tensor(hidden_states, True)}")

        logits = self.lm_head(hidden_states)
        # logits = cuda_ext.matmul_half(hidden_states, self.lm_head_data, cublas = False)

        if self.config.debug: print(f" !! logits: {_describe_tensor(logits, True)}")

        logits = logits.float()
        logits = _move_tensor(logits, self.config.device_map.embed_tokens, "logits", self.config)
        return logits
