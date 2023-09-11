from model import ExLlama, ExLlamaCache, ExLlamaConfig
from flask import Flask, request
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob

# Directory containing config.json, tokenizer.model and safetensors file for the model
model_directory = "/mnt/str/models/llama-7b-4bit/"

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)

config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
print(f"Model loaded: {model_path}")

tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator

# Flask app

app = Flask(__name__)


# Inference with settings equivalent to the "precise" preset from the /r/LocalLLaMA wiki

@app.route('/infer_precise', methods=['POST'])
def inferContextP():
    print(request.form)
    prompt = request.form.get('prompt')

    generator.settings.token_repetition_penalty_max = 1.176
    generator.settings.token_repetition_penalty_sustain = config.max_seq_len
    generator.settings.temperature = 0.7
    generator.settings.top_p = 0.1
    generator.settings.top_k = 40
    generator.settings.typical = 0.0    # Disabled

    outputs = generator.generate_simple(prompt, max_new_tokens = 200)
    return outputs


# Inference with settings equivalent to the "creative" preset from the /r/LocalLLaMA wiki

@app.route('/infer_creative', methods=['POST'])
def inferContextC():
    print(request.form)
    prompt = request.form.get('prompt')

    generator.settings.token_repetition_penalty_max = 1.1
    generator.settings.token_repetition_penalty_sustain = config.max_seq_len
    generator.settings.temperature = 0.72
    generator.settings.top_p = 0.73
    generator.settings.top_k = 0        # Disabled
    generator.settings.typical = 0.0    # Disabled

    outputs = generator.generate_simple(prompt, max_new_tokens = 200)
    return outputs


# Inference with settings equivalent to the "sphinx" preset from the /r/LocalLLaMA wiki

@app.route('/infer_sphinx', methods=['POST'])
def inferContextS():
    print(request.form)
    prompt = request.form.get('prompt')

    generator.settings.token_repetition_penalty_max = 1.15
    generator.settings.token_repetition_penalty_sustain = config.max_seq_len
    generator.settings.temperature = 1.99
    generator.settings.top_p = 0.18
    generator.settings.top_k = 30
    generator.settings.typical = 0.0    # Disabled

    outputs = generator.generate_simple(prompt, max_new_tokens = 200)
    return outputs


# Start Flask app

host = "0.0.0.0"
port = 8004
print(f"Starting server on address {host}:{port}")

if __name__ == '__main__':
    from waitress import serve
    serve(app, host = host, port = port)
