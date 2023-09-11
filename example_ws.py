import asyncio
import websockets
import json
from sentencepiece import SentencePieceProcessor

from model import ExLlama, ExLlamaCache, ExLlamaConfig
from lora import ExLlamaLora
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import argparse
import torch
import sys
import os
import glob
import model_init

# Initialized from command line args by init()

model: ExLlama
cache: ExLlamaCache
config: ExLlamaConfig
generator: ExLlamaGenerator
tokenizer: ExLlamaTokenizer
max_cached_strings = 100
tokenizer_cache = {}


prompt_ids: torch.tensor
stop_strings: list
stop_tokens: list
held_text: str
max_stop_string: int
remaining_tokens: int

full_prompt: str
utilized_prompt: str
built_response: str

def cached_tokenize(text: str):
    global model, cache, config, generator, tokenizer
    global max_cached_strings, tokenizer_cache

    if text in tokenizer_cache:
        return tokenizer_cache[text]

    while len(tokenizer_cache) >= max_cached_strings:
        del tokenizer_cache[next(iter(tokenizer_cache))]  # Always removes oldest entry as of Python 3.7

    new_enc = tokenizer.encode(text)
    tokenizer_cache[text] = new_enc
    return new_enc

def begin_stream(prompt: str, stop_conditions: list, max_new_tokens: int, gen_settings: ExLlamaGenerator.Settings):
    global model, cache, config, generator, tokenizer
    global stop_strings, stop_tokens, prompt_ids, held_text, max_stop_string, remaining_tokens
    global full_prompt, utilized_prompt, built_response

    # Tokenize prompt and limit length to allow prompt and (max) new tokens within max sequence length

    max_input_tokens = model.config.max_seq_len - max_new_tokens
    input_ids = cached_tokenize(prompt)
    input_ids = input_ids[:, -max_input_tokens:]
    prompt_ids = input_ids

    full_prompt = prompt
    utilized_prompt = tokenizer.decode(prompt_ids)[0]
    built_response = ""

    remaining_tokens = max_new_tokens

    # Settings

    stop_strings = []
    stop_tokens = []
    for t in stop_conditions:
        if isinstance(t, int): stop_tokens += [t]
        if isinstance(t, str): stop_strings += [t]

    held_text = ""

    max_stop_string = 2
    for ss in stop_strings:
        max_stop_string = max(max_stop_string, get_num_tokens(ss) + 2)

    generator.settings = gen_settings

    # Start generation

    generator.gen_begin_reuse(input_ids)

def stream():
    global model, cache, config, generator, tokenizer
    global stop_strings, stop_tokens, prompt_ids, held_text, max_stop_string, remaining_tokens
    global full_prompt, utilized_prompt, built_response

    # Check total response length

    if remaining_tokens == 0:
        return held_text, True, full_prompt + built_response, utilized_prompt + built_response, built_response
    remaining_tokens -= 1

    # Generate

    old_tail = tokenizer.decode(generator.sequence_actual[:, -max_stop_string:])[0]
    next_token = generator.gen_single_token()

    # End on stop token

    if next_token in stop_tokens:
        return held_text, True, full_prompt + built_response, utilized_prompt + built_response, built_response

    # Get new text

    new_tail = tokenizer.decode(generator.sequence_actual[:, -(max_stop_string + 1):])[0]
    added_text = new_tail[len(old_tail):]
    held_text += added_text

    # Hold text if it's part of a stop condition, end if it's a full stop condition

    partial_ss = False
    for ss in stop_strings:

        # Check if held_text fully contains stop string

        position = held_text.find(ss)
        if position != -1:
            built_response += held_text[:position]
            return held_text[:position], True, full_prompt + built_response, utilized_prompt + built_response, built_response

        # Check if end of held_text overlaps with start of stop string

        overlap = 0
        for j in range(1, min(len(held_text), len(ss)) + 1):
            if held_text[-j:] == ss[:j]: overlap = j
        if overlap > 0: partial_ss = True

    # Return partial result

    if partial_ss:
        return "", False, full_prompt + built_response, utilized_prompt + built_response, built_response

    stream_text = held_text
    held_text = ""
    built_response += stream_text
    return stream_text, False, full_prompt, utilized_prompt, built_response

def leftTrimTokens(text: str, desiredLen: int):

    encodedText = tokenizer.encode(text)
    if encodedText.shape[-1] <= desiredLen:
        return text
    else:
        return tokenizer.decode(encodedText[:, -desiredLen:])[0]

def oneshot_generation(prompt: str, stop_conditions: list, max_new_tokens: int, gen_settings: ExLlamaGenerator.Settings):

    begin_stream(prompt, stop_conditions, max_new_tokens, gen_settings)
    response = ""
    while True:
        _, eos, _, _, _ = stream()
        if eos: break

    return full_prompt + built_response, utilized_prompt + built_response, built_response


def get_num_tokens(text: str):

    return cached_tokenize(text).shape[-1]




# Websocket server
async def estimateToken(request, ws):
    text = request["text"]
    numTokens=get_num_tokens(text)
    return numTokens# return number of tokens in int

async def oneShotInfer(request, ws):
    stopToken = request["stopToken"]
    fullContext = request["text"]
    maxNew = int(request["maxNew"])
    top_p = float(request["top_p"])
    top_k = int(request["top_k"])
    temp = float(request["temp"])
    rep_pen = float(request["rep_pen"])
    sc = [tokenizer.eos_token_id]
    sc.append(stopToken)

    gs = ExLlamaGenerator.Settings()
    gs.top_k = top_k
    gs.top_p = top_p
    gs.temperature = temp
    gs.token_repetition_penalty_max = rep_pen

    full_ctx, util_ctx, response = oneshot_generation(prompt=fullContext, stop_conditions=sc, max_new_tokens=maxNew, gen_settings=gs)

    return full_ctx, util_ctx, response# return requested prompt/context, pruned prompt/context(eg. prunedctx+maxNew=4096), model generated response, not including prompt

async def streamInfer(request, ws):
    stopToken = [tokenizer.eos_token_id]
    stopToken += request["stopToken"].split(',')
    prompt = request["text"]
    maxNew = int(request["maxNew"])
    top_p = float(request["top_p"])
    top_k = int(request["top_k"])
    temp = float(request["temp"])
    rep_pen = float(request["rep_pen"])
    gs = ExLlamaGenerator.Settings()
    gs.top_k = top_k
    gs.top_p = top_p
    gs.temperature = temp
    gs.token_repetition_penalty_max = rep_pen
    begin_stream(prompt, stopToken, maxNew, gs)
    while True:
        chunk, eos, x, y, builtResp = stream()
        await ws.send(json.dumps({'action':request["action"],
                                  'request_id':request['request_id'],
                                  'utilContext':utilized_prompt + builtResp, 
                                  'response':builtResp}))
        if eos: break
    return utilized_prompt + built_response,builtResp


async def main(websocket, path):
    async for message in websocket:
        #try:
            request = json.loads(message)
            reqID = request["request_id"]
            action = request["action"]

            if action == "estimateToken":
                response = await estimateToken(request, websocket)
                await websocket.send(json.dumps({'action':action, 'request_id':reqID, 'response':response}))

            elif action == "echo":
                await websocket.send(json.dumps({'action':action, 'request_id':reqID}))

            elif action == "oneShotInfer":
                fctx, utlctx, res = await oneShotInfer(request, websocket)
                await websocket.send(json.dumps({'action':action, 'request_id':reqID,'utilContext':utlctx, 'response':res}))
            
            elif action == "leftTrim":
                prompt = request["text"]
                desiredLen = int(request["desiredLen"])
                processedPrompt = leftTrimTokens(prompt, desiredLen)
                await websocket.send(json.dumps({'action':action, 'request_id':reqID, 'response':processedPrompt}))

            else:
                utlctx, builtResp= await streamInfer(request, websocket)
                await websocket.send(json.dumps({'action':action, 'request_id':reqID,'utilContext':utlctx, 'response':builtResp+'</s>'}))



        #except Exception as e:
            #print({"error": str(e)})

model_directory = "./models/Llama-2-70B-chat-GPTQ/"

tokenizer_path = os.path.join(model_directory, "tokenizer.model")
model_config_path = os.path.join(model_directory, "config.json")
st_pattern = os.path.join(model_directory, "*.safetensors")
model_path = glob.glob(st_pattern)[0]
esTokenizer = SentencePieceProcessor(model_file = tokenizer_path)
config = ExLlamaConfig(model_config_path)               # create config from config.json
config.set_auto_map('17.615,18.8897')
config.model_path = model_path                          # supply path to model weights file

model = ExLlama(config)                                 # create ExLlama instance and load the weights
print(f"Model loaded: {model_path}")

tokenizer = ExLlamaTokenizer(tokenizer_path)            # create tokenizer from tokenizer model file
cache = ExLlamaCache(model)                             # create cache for inference
generator = ExLlamaGenerator(model, tokenizer, cache)   # create generator
start_server = websockets.serve(main, "0.0.0.0", 8080)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
