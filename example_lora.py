from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
from lora import ExLlamaLora
import os, glob
import torch

# Directory containt model, tokenizer, generator

model_directory = "/mnt/str/models/_test_models/Neko-Institute-of-Science_LLaMA-7B-4bit-128g/"

# Directory containing LoRA config and weights

lora_directory = "/mnt/str/models/_test_loras/tloen_alpaca-lora-7b/"

# Locate files we need within those directories

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)

lora_config_path = os.path.join(lora_directory, "adapter_config.json")
lora_path = os.path.join(lora_directory, "adapter_model.bin")

# Create config, model, tokenizer and generator

config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

# Load LoRA

lora = ExLlamaLora(model, lora_config_path, lora_path)

# Configure generator

generator.settings.token_repetition_penalty_max = 1.2
generator.settings.temperature = 0.65
generator.settings.top_p = 0.4
generator.settings.top_k = 0
generator.settings.typical = 0.0

# Alpaca prompt

prompt = \
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n" \
    "\n" \
    "### Instruction:\n" \
    "List five colors in alphabetical order.\n" \
    "\n" \
    "### Response:"

# Generate with LoRA

print(" --- LoRA ----------------- ")
print("")

generator.lora = lora
torch.manual_seed(1337)
output = generator.generate_simple(prompt, max_new_tokens = 200)
print(output)

# Generate without LoRA

print("")
print(" --- No LoRA -------------- ")
print("")

generator.lora = None
torch.manual_seed(1337)
output = generator.generate_simple(prompt, max_new_tokens = 200)
print(output)

