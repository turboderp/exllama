from model import ExLlamaConfig, Ex4bitLinear
import torch
import json
from safetensors.torch import load_file as safe_load_file
from torch import load as load_file

class ExLlamaLora:

    lora_config_path: str
    lora_path: str
    lora_r: int
    lora_alpha: float
    lora_scaling: float
    config: ExLlamaConfig
    tensors: dict[torch.tensor]
    bias_ignored: bool

    def __init__(self, model, lora_config_path, lora_path):

        self.lora_config_path = lora_config_path
        self.lora_path = lora_path
        self.model = model
        self.config = model.config
        self.tensors = {}
        self.bias_ignored = False

        # Grab relevant items from LoRA config

        with open(lora_config_path) as f:
            read_config = json.load(f)

        self.lora_r = read_config["r"]
        self.lora_alpha = float(read_config["lora_alpha"])
        self.lora_scaling = self.lora_alpha / self.lora_r

        if "fan_in_fan_out" in read_config and read_config["fan_in_fan_out"]:
            raise ValueError(" ## Error: fan_in_fan_out mode not supported.")

        # Load LoRA weights

        if self.lora_path.endswith(".safetensors"):
            f = safe_load_file(self.lora_path, device = "cpu")
        else:
            f = load_file(self.lora_path, map_location = "cpu")

        for key in f.keys():
            tensor = f[key]

            # Find target

            i = key.find("model.layers.")
            if i == -1: raise ValueError(f" ## Error: unsupported layer in {self.lora_path}: {key}")

            target_key = key[i:]
            ks = target_key.split(".")
            decoder_idx = int(ks[2])
            decoder_part = ks[3]
            decoder_layer = ks[4]
            lora_half = ks[5]

            if lora_half == "bias":
                epsilon = 1e-6
                if torch.max(tensor) > epsilon or torch.max(tensor) < -epsilon:
                    raise ValueError(f" ## Error: unsupported bias target {self.lora_path}: {key}")
                self.bias_ignored = True
                continue

            target_module = self.model.layers[decoder_idx]
            if decoder_part == "self_attn": target_module = target_module.self_attn
            elif decoder_part == "mlp": target_module = target_module.mlp
            else: raise ValueError(f" ## Error: unsupported layer in {self.lora_path}: {key}")

            if   decoder_layer == "q_proj": target_module = target_module.q_proj
            elif decoder_layer == "k_proj": target_module = target_module.k_proj
            elif decoder_layer == "v_proj": target_module = target_module.v_proj
            elif decoder_layer == "o_proj": target_module = target_module.o_proj
            elif decoder_layer == "gate_proj": target_module = target_module.gate_proj
            elif decoder_layer == "up_proj": target_module = target_module.up_proj
            elif decoder_layer == "down_proj": target_module = target_module.down_proj
            else: raise ValueError(f" ## Error: unsupported layer in {self.lora_path}: {key}")

            # Check that shape is compatible

            assert isinstance(target_module, Ex4bitLinear)

            if lora_half == "lora_A":
                in_features = tensor.shape[1]
                out_features = None
            elif lora_half == "lora_B":
                in_features = None
                out_features = tensor.shape[0]
            else: raise ValueError(f" ## Error: unsupported layer in {self.lora_path}: {key}")

            if (in_features and in_features != target_module.in_features) or (out_features and out_features != target_module.out_features):
                raise ValueError(f" ## Error: incompatible tensor shape in {self.lora_path}: {key}")

            # For efficiency, transpose adapter instead of transposing state during inference

            tensor = tensor.T.contiguous()

            # Pre-scale

            if lora_half == "lora_B" and self.lora_scaling != 1.0: tensor.mul_(self.lora_scaling)

            # Check that dtype is compatible, or convert

            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)

            elif tensor.dtype == torch.float32:
                tensor = tensor.to(torch.float16)

            elif tensor.dtype == torch.float16:
                pass

            else: raise ValueError(f" ## Error: unsupported tensor dtype in {self.lora_path}")

            # Move to target device

            device = self.config.device_map.map(target_key)
            tensor = tensor.to(device, non_blocking = True)

            # Store adapter tensor

            self.tensors[target_key] = tensor
