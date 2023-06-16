from flask import Flask, request
import json 
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob
import time

model_directory =  "model/30B-Lazarus-GPTQ4bit/"

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]

config = ExLlamaConfig(model_config_path)               # create config from config.json
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file

cache = ExLlamaCache(model)                             # create cache for inference

# Configure generator



print("model loaded")
app = Flask(__name__)

generatorP = ExLlamaGenerator(model, tokenizer, cache)   # create generator
#generatorP.disallow_tokens([tokenizer.eos_token_id])
generatorP.settings.token_repetition_penalty_max = 1.176
generatorP.settings.temperature = 0.7
generatorP.settings.top_p = 0.1
generatorP.settings.top_k = 40
generatorP.settings.typical = 0.5



generatorC = ExLlamaGenerator(model, tokenizer, cache)   # create generator
#generatorC.disallow_tokens([tokenizer.eos_token_id])
generatorC.settings.token_repetition_penalty_max = 1.1
generatorC.settings.temperature = 0.72
generatorC.settings.top_p = 0.73
generatorC.settings.top_k = 0
generatorC.settings.typical = 0.5


generatorS = ExLlamaGenerator(model, tokenizer, cache)   # create generator
#generatorS.disallow_tokens([tokenizer.eos_token_id])
generatorS.settings.token_repetition_penalty_max = 1.15
generatorS.settings.temperature = 1.99
generatorS.settings.top_p = 0.18
generatorS.settings.top_k = 30
generatorS.settings.typical = 0.5

@app.route('/infer_precise', methods=['POST'])
def inferContextP():
    print(request.form)
    prompt = request.form.get('prompt')

    outputs = generatorP.generate_simple(prompt, max_new_tokens = 200)
    return outputs

@app.route('/infer_creative', methods=['POST'])
def inferContextC():
    print(request.form)
    prompt = request.form.get('prompt')

    outputs = generatorC.generate_simple(prompt, max_new_tokens = 200)
    return outputs


@app.route('/infer_sphinx', methods=['POST'])
def inferContextS():
    print(request.form)
    prompt = request.form.get('prompt')

    outputs = generatorS.generate_simple(prompt, max_new_tokens = 200)
    return outputs




if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8004)
