from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import torch

# Just a quick test to see if we are getting anything sensible out of the model.

torch.set_grad_enabled(False)
torch.cuda._lazy_init()

tokenizer_model_path = "/mnt/str/models/llama-7b-4bit-128g/tokenizer.model"
model_config_path = "/mnt/str/models/llama-7b-4bit-128g/config.json"
model_path = "/mnt/str/models/llama-7b-4bit-128g/llama-7b-4bit-128g.safetensors"

# tokenizer_model_path = "/mnt/str/models/llama-13b-4bit-128g/tokenizer.model"
# model_config_path = "/mnt/str/models/llama-13b-4bit-128g/config.json"
# model_path = "/mnt/str/models/llama-13b-4bit-128g/llama-13b-4bit-128g.safetensors"
#
# tokenizer_model_path = "/mnt/str/models/llama-30b-4bit-128g/tokenizer.model"
# model_config_path = "/mnt/str/models/llama-30b-4bit-128g/config.json"
# model_path = "/mnt/str/models/llama-30b-4bit-128g/llama-30b-4bit-128g.safetensors"

# tokenizer_model_path = "/mnt/str/models/llama-30b-4bit-128g-act/tokenizer.model"
# model_config_path = "/mnt/str/models/llama-30b-4bit-128g-act/config.json"
# model_path = "/mnt/str/models/llama-30b-4bit-128g-act/llama-30b-4bit-128g.safetensors"

config = ExLlamaConfig(model_config_path)
config.model_path = model_path
# config.attention_method = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP
# config.matmul_method = ExLlamaConfig.MatmulMethod.QUANT_ONLY
config.max_seq_len = 2048
model = ExLlama(config)
cache = ExLlamaCache(model)

tokenizer = ExLlamaTokenizer(tokenizer_model_path)
generator = ExLlamaGenerator(model, tokenizer, cache)
generator.settings.token_repetition_penalty_max = 1.2
generator.settings.token_repetition_penalty_sustain = 20
generator.settings.token_repetition_penalty_decay = 50

prompt = \
"On 19 February 1952, Headlam became senior air staff officer (SASO) at Eastern Area Command in Penrith, New South " \
"Wales. During his term as SASO, the RAAF began re-equipping with English Electric Canberra jet bombers and CAC " \
"Sabre jet fighters. The Air Force also underwent a major organisational change, as it transitioned from a " \
"geographically based command-and-control system to one based on function, resulting in the establishment of Home " \
"(operational), Training, and Maintenance Commands. Eastern Area Command, considered a de facto operational " \
"headquarters owing to the preponderance of combat units under its control, was reorganised as Home Command in " \
"October 1953. Headlam was appointed an Officer of the Order of the British Empire (OBE) in the 1954 New Year " \
"Honours for his \"exceptional ability and devotion to duty\". He was promoted to acting air commodore in May. His " \
"appointment as aide-de-camp to Queen Elizabeth II was announced on 7 October 1954."

gen_tokens = 200
text = generator.generate_simple(prompt, max_new_tokens = gen_tokens)
print(text)

