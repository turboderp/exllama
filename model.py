import torch
from torch import nn
import torch.nn.functional as F
from safetensors import safe_open
import quant_util
import json
import math
from enum import Enum

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

        QUANT_ONLY = 1
        SWITCHED = 2  # Best
        PYTORCH_ONLY = 3


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

        # Required settings

        self.model_path = None
        self.groupsize = None  # Group size used for quantized model, specify -1 for v1 model

        # Optional settings

        self.max_seq_len = 2048  # Reduce to save memory. Can also be increased, but the pretrained models produce degenerate output after 2048 tokens in any case. Should be possible to finetune for longer sequence lengths.
        self.is_v1_model = False  # TODO: Sort out v1 models?
        self.attention_method = self.AttentionMethod.PYTORCH_SCALED_DP
        self.matmul_method = self.MatmulMethod.SWITCHED
        self.device_map = ExLlamaDeviceMap(self.num_hidden_layers)


# 4-bit linear layer implementation

class Ex4bitLinear(nn.Module):

    def __init__(self, config, in_features, out_features, has_bias, tensors, key):
        super().__init__()

        self.config = config

        self.in_features = in_features
        self.out_features = out_features
        self.bits = 4  # quant_cuda provides functions for 2 and 3 bits as well, but they will be unsupported for now
        self.maxq = 2 ** self.bits - 1
        self.has_bias = has_bias

        self.groupsize = self.config.groupsize if self.config.groupsize != -1 else in_features

        if self.config.is_v1_model:

            # TODO: v1 models are currently untested. Test some. Maybe? Are they still relevant?

            self.register_buffer('zeros', tensors[key + ".zeros"])
            self.register_buffer('scales', tensors[key + ".scales"])

        else:

            self.register_buffer('qzeros', tensors[key + ".qzeros"])
            self.register_buffer('scales', tensors[key + ".scales"])

        if self.has_bias: self.register_buffer('bias', tensors[key + ".bias"])

        self.register_buffer('qweight', tensors[key + ".qweight"])

    def forward(self, x):

        zeros = self.qzeros if not self.config.is_v1_model else self.zeros

        if torch.is_grad_enabled():

            out = quant_util.ExAutogradMatmul4bitCuda.apply(x, self.qweight, self.scales, zeros, self.groupsize, self.bits, self.maxq)

        else:

            if self.config.matmul_method == ExLlamaConfig.MatmulMethod.QUANT_ONLY: auto_switch_thd = -1
            elif self.config.matmul_method == ExLlamaConfig.MatmulMethod.SWITCHED: auto_switch_thd = optimal_switch_thd
            elif self.config.matmul_method == ExLlamaConfig.MatmulMethod.PYTORCH_ONLY: auto_switch_thd = 0
            else: raise ValueError("Wut?")

            out = quant_util.matmul4bit(x, self.qweight, self.scales, zeros, self.groupsize, auto_switch_thd = auto_switch_thd)

        if self.has_bias: out += self.bias
        return out


# Llama MLP

class ExLlamaMLP(nn.Module):

    def __init__(self, config, tensors, key):
        super().__init__()

        self.config = config

        self.gate_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.intermediate_size, False, tensors, key + ".gate_proj")
        self.down_proj = Ex4bitLinear(config, self.config.intermediate_size, self.config.hidden_size, False, tensors, key + ".down_proj")
        self.up_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.intermediate_size, False, tensors, key + ".up_proj")

        self.act_fn = nn.SiLU()


    def forward(self, x):

        y = self.gate_proj(x)
        y = self.act_fn(y)
        y *= self.up_proj(x)
        y = self.down_proj(y)
        return y


# RMS Layer norm. TODO: Test if upcasting is necessary and/or figure out if the extra allocation matters

class ExLlamaRMSNorm(nn.Module):

    def __init__(self, config, tensors, key):
        super().__init__()

        self.config = config
        self.variance_epsilon = self.config.rms_norm_eps
        self.weight = nn.Parameter(tensors[key])


    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim = True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        if self.weight.dtype in [torch.float16, torch.bfloat16]: hidden_states = hidden_states.to(self.weight.dtype)
        return self.weight * hidden_states


# Llama attention

class ExLlamaAttention(nn.Module):

    def __init__(self, config, tensors, key, sin, cos, index):
        super().__init__()

        self.config = config
        self.sin = sin
        self.cos = cos
        self.index = index

        self.q_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".q_proj")
        self.k_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".k_proj")
        self.v_proj = Ex4bitLinear(config, self.config.hidden_size, self.config.num_attention_heads * self.config.head_dim, False, tensors, key + ".v_proj")
        self.o_proj = Ex4bitLinear(config, self.config.num_attention_heads * self.config.head_dim, self.config.hidden_size, False, tensors, key + ".o_proj")


    def forward(self, hidden_states, cache, attention_mask):

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
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)
            attn_output = attn_output.transpose(1, 2)
            del attn_weights

        # -- Scaled dot-product attention from PyTorch 2, should be comparable to xformers (?)

        elif self.config.attention_method == ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP:

            attn_output = F.scaled_dot_product_attention(query_states, key_states, value_states, attn_mask = attention_mask, is_causal = False)
            attn_output = attn_output.transpose(1, 2)

        else: raise ValueError("Wut?")

        # Output projection

        attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output


class ExLlamaDecoderLayer(nn.Module):

    def __init__(self, config, tensors, key, index, sin, cos):
        super().__init__()

        self.config = config
        self.index = index

        self.self_attn = ExLlamaAttention(self.config, tensors, key + ".self_attn", sin, cos, self.index)
        self.mlp = ExLlamaMLP(self.config, tensors, key + ".mlp")

        self.input_layernorm = ExLlamaRMSNorm(self.config, tensors, key + ".input_layernorm.weight")
        self.post_attention_layernorm = ExLlamaRMSNorm(self.config, tensors, key + ".post_attention_layernorm.weight")


    def forward(self, hidden_states, cache, attention_mask):

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cache, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


# Persistent cache for inference. Allocate the whole thing up front.

class ExLlamaCache:

    def __init__(self, model, batch_size = 1, max_seq_len = -1):

        self.model = model
        self.config = self.model.config
        self.max_seq_len = max_seq_len if max_seq_len != -1 else self.config.max_seq_len
        self.batch_size = batch_size

        self.key_states = []
        self.value_states = []
        self.current_seq_len = 0

        # Preallocate full-length cache

        for i in range(self.config.num_hidden_layers):

            p_key_states = torch.zeros(self.batch_size, self.config.num_attention_heads, self.max_seq_len, self.config.head_dim, dtype = torch.float16, device = self.model.config.device_map.layers[i])
            p_value_states = torch.zeros(self.batch_size, self.config.num_attention_heads, self.max_seq_len, self.config.head_dim, dtype = torch.float16, device = self.model.config.device_map.layers[i])

            self.key_states.append(p_key_states)
            self.value_states.append(p_value_states)


# Device map for the model. Currently untested, but should allow for each individual layers to reside on any device.
# Although the quant stuff probably only works on CUDA. For now.

class ExLlamaDeviceMap:

    def __init__(self, num_layers):

        self.num_layers = num_layers

        self.embed_tokens = "cpu"  # Embedding table on CPU saves 400 MB on the 30B model with no measurable impact on performance
        self.lm_head = "cuda:0"
        self.norm = "cuda:0"
        self.layers = ["cuda:0"] * self.num_layers


    def get_layers_devs(self):

        return list(set(self.layers))


    def map(self, key):

        if key.startswith("lm_head."): return self.lm_head
        if key.startswith("model.embed_tokens."): return self.embed_tokens
        if key.startswith("model.norm."): return self.norm

        if key.startswith("model.layers."):
            num = int(key.split(".")[2])
            return self.layers[num]

        raise ValueError("Unknown key: " + key)


class ExLlama(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.eval()

        self.config = config

        # Load model weights

        tensors = {}
        with safe_open(self.config.model_path, framework="pt", device="cpu") as f:
            for key in f.keys():

                if key.endswith("_proj.bias"): continue  # Skip loading unused, empty bias tensors
                if key.endswith(".rotary_emb.inv_freq"): continue  # This is always precomputed during init anyway

                device = self.config.device_map.map(key)
                tensor = f.get_tensor(key)

                if key.endswith(".scales"): tensor = tensor.half()

                tensor = tensor.to(device, non_blocking = True)
                tensors[key] = tensor
                # print(key + " -> " + device)

        # Head

        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias = False, device = "meta")
        self.lm_head.weight = nn.Parameter(tensors["lm_head.weight"])

        # Token embeddings

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size, self.config.pad_token_id, device = "meta")
        self.embed_tokens.weight = nn.Parameter(tensors["model.embed_tokens.weight"])

        # Norm

        self.norm = ExLlamaRMSNorm(self.config, tensors, "model.norm.weight")

        # Prepare position embeddings for max seq length

        self.sincos = {}
        for device in self.config.device_map.get_layers_devs():

            inv_freq = 1.0 / (self.config.rotary_embedding_base ** (torch.arange(0, self.config.head_dim, 2, device = device).float() / self.config.head_dim))
            t = torch.arange(self.config.max_seq_len, device = device, dtype = torch.float32)
            freqs = torch.einsum("i,j->ij", t, inv_freq)
            emb = torch.cat((freqs, freqs), dim = -1)

            sin = emb.sin()[None, None, :, :].half()
            cos = emb.cos()[None, None, :, :].half()

            self.sincos[device] = (sin, cos)

        # Layers

        modules = []
        for i in range(self.config.num_hidden_layers):
            device = self.config.device_map.layers[i]
            sin, cos = self.sincos[device]
            layer = ExLlamaDecoderLayer(self.config, tensors, f"model.layers.{i}", i, sin, cos)
            modules.append(layer)

        self.layers = nn.ModuleList(modules)


    def forward(self, input_ids, cache, last_id_only = True, preprocess_only = False):

        batch_size, seq_len = input_ids.shape
        past_len = cache.current_seq_len

        # Build attention mask on first device, copy to others if necessary

        attn_masks = {}
        devs = self.config.device_map.get_layers_devs()

        attn_mask = torch.zeros(batch_size, 1, seq_len, seq_len + past_len, dtype = torch.float16, device = devs[0])

        if seq_len > 1:
            attn_mask = torch.zeros(batch_size, 1, seq_len, past_len + seq_len, dtype = torch.float16, device = devs[0])
            attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), torch.finfo(torch.float16).min))
            attn_mask[:, :, : seq_len - 1, past_len + 1: past_len + seq_len] = attn_mask_triu

        attn_masks[devs[0]] = attn_mask
        for device in devs[1:]: attn_masks[device] = attn_mask.to(device)

        # Embeddings
        # TODO: Allow passing input embeddings instead of IDs

        hidden_states = self.embed_tokens(input_ids.to(self.config.device_map.embed_tokens))

        # Decoder layers

        for idx, decoder_layer in enumerate(self.layers):
            device = self.config.device_map.layers[idx]
            hidden_states = hidden_states.to(device)
            hidden_states = decoder_layer(hidden_states, cache, attn_masks[device])

        cache.current_seq_len += seq_len

        # Early exit when we don't need logits

        if preprocess_only: return None

        # Norm

        hidden_states = hidden_states.to(self.config.device_map.norm)
        hidden_states = self.norm(hidden_states)

        # Head

        if last_id_only: hidden_states = hidden_states[:, -1:, :]

        hidden_states = hidden_states.to(self.config.device_map.lm_head)
        logits = self.lm_head(hidden_states)

        return logits.to(self.config.device_map.embed_tokens).float()

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