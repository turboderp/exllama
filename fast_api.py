import glob
import time
import asyncio
import uvicorn
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
# exllama imports:
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import argparse
import torch
import sys
import os


# [init torch]:
torch.set_grad_enabled(False)
torch.cuda._lazy_init()

# [Parse arguments]:
parser = argparse.ArgumentParser(description = "Simple FastAPI wrapper for ExLlama")

parser.add_argument("-t", "--tokenizer", type = str, help = "Tokenizer model path",default = None)
parser.add_argument("-c", "--config", type = str, help = "Model config path (config.json)", default = None)
parser.add_argument("-d", "--directory", type = str, help = "Path to directory containing config.json, model.tokenizer and * .safetensors")
#...

# Add the rest of the args here (at least the ones not sent via web request)
# Also.. start implementing the ones from GenerateRequest

args = parser.parse_args()

# Directory check:
if args.directory is not None:
    args.tokenizer = os.path.join(args.directory, "tokenizer.model")
    args.config = os.path.join(args.directory, "config.json")
    st_pattern = os.path.join(args.directory, "*.safetensors")
    st = glob.glob(st_pattern)
    if len(st) == 0:
        print(f" !! No files matching {st_pattern}")
        sys.exit()
    if len(st) > 1:
        print(f" !! Multiple files matching {st_pattern}")
        sys.exit()
    args.model = st[0]
else:
    if args.tokenizer is None or args.config is None or args.model is None:
        print(" !! Please specify either -d or all of -t, -c and -m")
        sys.exit()

# -m model sent, but no -t or -c, we can construct it!
# python fast_api.py -m /home/nap/llm_models/koala-13B-HF-4bit/koala13B-4bit-128g.safetensors
#if args.tokenizer is None and args.model:
#    directory = os.path.dirname(args.model)
#    args.tokenizer = os.path.join(directory, "tokenizer.model")
#    args.config = os.path.join(directory, "config.json")
    
# Validate we have model params:
#if args.model is None or args.tokenizer is None or args.config is None:
#    print("Missing model params, send them in as flags!")
#    sys.exit()
#-------


# Instantiate model and generator
config = ExLlamaConfig(args.config)
config.model_path = args.model
config.max_seq_len = 2048
#config.attention_method = ExLlamaConfig.AttentionMethod.PYTORCH_SCALED_DP
#config.matmul_method = ExLlamaConfig.MatmulMethod.SWITCHED

model = ExLlama(config)
cache = ExLlamaCache(model)
tokenizer = ExLlamaTokenizer(args.tokenizer)
generator = ExLlamaGenerator(model, tokenizer, cache)

# Some feedback
print(f" -- Loading model")
print(f" -- Tokenizer: {args.tokenizer}")
print(f" -- Model config: {args.config}")
print(f" -- Model: {args.model}")
print(f" -- Groupsize (inferred): {model.config.groupsize if model.config.groupsize is not None else 'None'}")
#-------


#'''
# Set these from GenerateRequest, per request. (after we get everything working)
generator.settings = ExLlamaGenerator.Settings()
generator.settings.temperature = 0.95
generator.settings.top_k = 20
generator.settings.top_p = 0.65
generator.settings.min_p = 0.06
generator.settings.token_repetition_penalty_max = 1.15
generator.settings.token_repetition_penalty_sustain = 256
generator.settings.token_repetition_penalty_decay = generator.settings.token_repetition_penalty_sustain // 2
#'''


# Setup FastAPI:
app = FastAPI()
semaphore = asyncio.Semaphore(1)

# I need open CORS for my setup, you may not!!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#-------


# Booyah!
@app.get("/")
def hellow_world(q: Union[str, None] = None):
    return {"wintermute": "Hellow World!", "q": q}


# fastapi_chat.html uses this to check what model is being used.
# (My Webserver uses this to check if my LLM is running):
@app.get("/check")
def check():
    return { "model": args.model }


class GenerateRequest(BaseModel):
    message: str
    prompt: Optional[str] = None
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    # options:
    stream: Optional[bool] = True
    log: Optional[bool] = True
    evl: Optional[int] = 1
				        

@app.post("/generate")
async def stream_data(req: GenerateRequest):
    while True:
        try:
            # Attempt to acquire the semaphore without waiting, in a loop...
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            break
        except asyncio.TimeoutError:
            print("Server is busy")
            await asyncio.sleep(1)
    
    try:
        # start timer:
        t0 = time.time()

        # place user message into prompt:
        if req.prompt:
            _MESSAGE = req.prompt.replace("{user_input}", req.message)
        else:
            _MESSAGE = req.message
        #print(_MESSAGE)

        # copy of generate_simple() so that I could yield each token for streaming without having to change generator.py and make merging updates a nightmare:
        async def generate_simple(prompt, settings = generator.Settings(), max_new_tokens = 128):
            t0 = time.time()
            new_text = ""
            last_text = ""
            _full_answer = ""

            generator.end_beam_search()
            generator.settings = settings

            ids = tokenizer.encode(prompt)
            generator.gen_begin(ids)

            for i in range(max_new_tokens):
                token = generator.gen_single_token()
                if token.item() == tokenizer.eos_token_id: break
                text = tokenizer.decode(generator.sequence[0])
                new_text = text[len(_MESSAGE):]

                # Get new token by taking difference from last response:
                new_token = new_text.replace(last_text, "")
                last_text = new_text

                print(new_token, end="", flush=True)
                #if req.stream:
                yield new_token

            # all done:
            generator.end_beam_search() 
            _full_answer = new_text

            # get num new tokens:
            prompt_tokens = tokenizer.encode(_MESSAGE)
            prompt_tokens = len(prompt_tokens[0])
            new_tokens = tokenizer.encode(_full_answer)
            new_tokens = len(new_tokens[0])

            # calc tokens/sec:
            t1 = time.time()
            _sec = t1-t0
            _tokens_sec = new_tokens/(_sec)

            print(f"full answer: {_full_answer}")

            print(f"Output generated in {_sec} ({_tokens_sec} tokens/s, {new_tokens}, context {prompt_tokens})")

        if req.stream:
            return StreamingResponse(generate_simple(_MESSAGE))
        else:
            return { generate_simple(_MESSAGE) }

    except Exception as e:
        return {'response': f"Exception while processing request: {e}"}

    finally:
        semaphore.release()


@app.post("/backup_generate")
async def stream_data(req: GenerateRequest):
    while True:
        try:
            # Attempt to acquire the semaphore without waiting, in a loop...
            await asyncio.wait_for(semaphore.acquire(), timeout=0.1)
            break
        except asyncio.TimeoutError:
            print("Server is busy")
            await asyncio.sleep(1)
    
    try:
        # start timer:
        t0 = time.time()

        # place user message into prompt:
        if req.prompt:
            _MESSAGE = req.prompt.replace("{user_input}", req.message)
        else:
            _MESSAGE = req.message
        #print(_MESSAGE)

        if req.stream:
            # Generate with streaming:
            async def gen():
                t0 = time.time()
                new_text = ""
                last_text = ""
                _full_answer = ""

                # Encode _MESSAGE and send to gen_begin:
                in_tokens = tokenizer.encode(_MESSAGE)
                num_in_tokens = in_tokens.shape[-1]  # Decode from here
                generator.gen_begin(in_tokens)
            
                # Prune context if it gets too big:
                expect_tokens = in_tokens.shape[-1] + req.max_new_tokens
                max_tokens = config.max_seq_len - expect_tokens
                if generator.gen_num_tokens() >= max_tokens:
                    generator.gen_prune_to(config.max_seq_len - expect_tokens - 256, tokenizer.newline_token_id) #   extra_prune = 256

                # Generate Tokens:
                generator.begin_beam_search()

                for i in range(req.max_new_tokens):
                
                    # Disallowing the end condition tokens seems like a clean way to force longer replies.
                    if i < 4: #min_response_tokens:
                        generator.disallow_tokens([tokenizer.newline_token_id, tokenizer.eos_token_id])
                    else:
                        generator.disallow_tokens(None)

                    # Get a token

                    gen_token = generator.beam_search()

                    # If token is EOS, replace it with newline before continuing

                    if gen_token.item() == tokenizer.eos_token_id:
                        generator.replace_last_token(tokenizer.newline_token_id)

                    # Decode the current line and print any characters added

                    num_in_tokens += 1
                    text = tokenizer.decode(generator.sequence_actual[:, -num_in_tokens:][0])
                    new_text = text[len(_MESSAGE):]

                    # Get new token by taking difference from last response:
                    new_token = new_text.replace(last_text, "")
                    last_text = new_text

                    print(new_token, end="", flush=True)
                    yield new_token

                    # break on newline_token_id so chatbort doesn't start writing for the user...
                    if gen_token.item() == tokenizer.newline_token_id:
                        break
                """
                for i in range(req.max_new_tokens):
                    gen_token = generator.gen_single_token()
                    token = gen_token
                    if gen_token.item() == tokenizer.eos_token_id:
                        token = torch.tensor([[tokenizer.newline_token_id]])

                    # accept new token into sequence:
                    generator.gen_accept_token(token)

                    # Decode response:
                    num_in_tokens += 1
                    text = tokenizer.decode(generator.sequence[:, -num_in_tokens:][0])
                    new_text = text[len(_MESSAGE):]

                    # Get new token by taking difference from last response:
                    new_token = new_text.replace(last_text, "")
                    last_text = new_text

                    print(new_token, end="", flush=True)
                    yield new_token

                    # break on newline_token_id so chatbort doesn't start writing for the user...
                    if gen_token.item() == tokenizer.newline_token_id:
                        break
                """

                generator.end_beam_search() 

                _full_answer = new_text

                # get num new tokens:
                prompt_tokens = tokenizer.encode(_MESSAGE)
                prompt_tokens = len(prompt_tokens[0])
                new_tokens = tokenizer.encode(_full_answer)
                new_tokens = len(new_tokens[0])

                # calc tokens/sec:
                t1 = time.time()
                _sec = t1-t0
                _tokens_sec = new_tokens/(_sec)

                print(f"full answer: {_full_answer}")

                print(f"Output generated in {_sec} ({_tokens_sec} tokens/s, {new_tokens}, context {prompt_tokens})")

            return StreamingResponse(gen())
        else:
            # No streaming, using generate_simple:
            text = generator.generate_simple(_MESSAGE, max_new_tokens=req.max_new_tokens)
            print(text)

            # remove prompt from response:
            new_text = text.replace(_MESSAGE,"")
            new_text = new_text.lstrip()
            #print(new_text)

            # get num new tokens:
            prompt_tokens = tokenizer.encode(_MESSAGE)
            prompt_tokens = len(prompt_tokens[0])
            new_tokens = tokenizer.encode(new_text)
            new_tokens = len(new_tokens[0])

            # calc tokens/sec:
            t1 = time.time()
            _sec = t1-t0
            _tokens_sec = new_tokens/(_sec)

            print(f"Output generated in {_sec} ({_tokens_sec} tokens/s, {new_tokens}, context {prompt_tokens})")

            # return response time here?
            return { new_text }

    except Exception as e:
        return {'response': f"Exception while processing request: {e}"}

    finally:
        semaphore.release()
#-------


if __name__ == "__main__":
    # [start fastapi]:
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7862
    )