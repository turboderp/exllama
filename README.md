# ExLlama

A rewrite of the HF transformers implementation of Llama with the following goals, among others:

* Designed for use with quantized weights
* Fast and memory-efficient inference (not just attention)
* Mapping across multiple devices
* Built-in (multi) LoRA support
* Companion library of funky sampling functions

Disclaimer: This is currently a preview of a work in progress. Or maybe a proof of concept. Either way any part of it
is subject to change.

## Hardware/software requirements

I am developing on an RTX 4090 and an RTX 3090-Ti. Both cards support the CUDA kernel, but there might be
incompatibilities with older cards. I have no way of testing that right now.

## Dependencies

This list might be incomplete:

* `torch` tested on 2.1.0 (nightly) with cu118
* `safetensors` 0.3.1
* `sentencepiece`
* `ninja`
* `flask` (only for the web UI)

## Linux/WSL prerequisites

    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118

## Windows prerequisites

To run on Windows (without WSL):

1. Install [MSVC 2022](https://visualstudio.microsoft.com/downloads/). You can choose to install the whole `Visual 
Studio 2022` IDE, or alternatively just the `Build Tools for Visual Studio 2022` package (make sure `Desktop
development with C++` is ticked in the installer), it doesn't really matter which.
2. Install the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/), choosing one of the CUDA
versions. I am developing on the nightly build, but the stable version should also work.
3. Install CUDA Toolkit, ([11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive) and 
[11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive) both seem to work, just make sure to match PyTorch's
Compute Platform version).
4. For best performance, enable Hardware Accelerated GPU Scheduling.

## How to

Install dependencies, clone repo and run benchmark:

    pip install safetensors sentencepiece ninja

    git clone https://github.com/turboderp/exllama
    cd exllama

    python test_benchmark_inference.py -d <path_to_model_files> -p -ppl

The CUDA extension is loaded at runtime so there's no need to install it separately. It will be compiled on the first
run and cached to `~/.cache/torch_extensions/` which could take a little while. If nothing happens at first, give it
a minute to compile.

Chatbot example:

    python test_chatbot.py -d <path_to_model_files> -un "Jeff" -p prompt_chatbort.txt

## Web UI

I made a simple web UI for it. Like the rest of the project, it's a work in progress. Don't look at the JavaScript,
it was mostly written by ChatGPT and it will haunt your dreams. But it sort of works, and it's kinda fun, especially
multibot mode:

![_screenshot.jpg](_screenshot.jpg)

To run it:

    pip install flask

    python webui/app.py -d <path_to_model_files>

Note that sessions are stored in `~/exllama_sessions/`. 

## Results so far

### New implementation
| Model    | Size | grpsz | act             | Seq. len.            | VRAM      | Prompt     | Best    | Worst   | Ppl  |
|----------|------|-------|-----------------|----------------------|-----------|------------|---------|---------|------|
| Llama    | 7B   | 128   | no              | 2,048 t              | 5,194 MB  | 10,460 t/s | 160 t/s | 133 t/s | 6.45 |
| Llama    | 13B  | 128   | no              | 2,048 t              | 9,127 MB  | 5,831 t/s  | 97 t/s  | 83 t/s  | 5.60 |
| Llama    | 30B  | 128   | no              | 2,048 t              | 20,795 MB | 2,481 t/s  | 46 t/s  | 39 t/s  | 4.60 |
| Llama    | 30B  | 128   | yes             | 2,048 t              | 20,795 MB | 2,343 t/s  | 44 t/s  | 37 t/s  | 4.55 |
| Llama    | 30B  | 32    | yes             | 1,550 t <sup>1</sup> | 21,486 MB | 2,308 t/s  | 40 t/s  | 36 t/s  | 4.52 |
| Koala    | 13B  | 128   | yes             | 2,048 t              | 9,127 MB  | 5,529 t/s  | 86 t/s  | 79 t/s  | 6.73 |
| WizardLM | 30B  | -     | no <sup>2</sup> | 2,048 t              | 20,199 MB | 2,313 t/s  | 44 t/s  | 39 t/s  | 5.75 |

<sup>1</sup> Can not achieve full sequence length without OoM (yet)  
<sup>2</sup> Not quite sure if this is act-order or not. Weights have no group index, at least   

All tests done on stock RTX 4090 / 12900K, running with a desktop environment, with a few other apps also using VRAM.

**"Prompt"** speed is inference over the sequence length listed minus 128 tokens. **"Worst"** is the average speed for
the last 128 tokens of the full context (worst case) and **"Best"** lists the speed for the first 128 tokens in an
empty sequence (best case.)

VRAM usage is as reported by PyTorch and does not include PyTorch's own overhead (CUDA kernels,
internal buffers etc.) This is somewhat unpredictable anyway. Best bet is to just optimize VRAM usage by the model,
probably aiming for 20 GB on a 24 GB GPU to ensure there is room for a desktop environment and all of Torch's
internals.

Perplexity is measured only to verify that the models are working. The dataset used is a particular, small sample from
WikiText, so scores are not necessarily comparable to other Llama benchmarks.

### Dual GPU results

Since many seem to be interested in running 65B models, I can confirm that this works with two 24 GB GPUs. The
following benchmarks are from a 4090 + 3090-Ti with `-gs 17.2,24`:

| Model    | Size | groupsize | act | Seq. len.            | VRAM      | Prompt  | Best   | Worst  | Ppl  |
|----------|------|-----------|-----|----------------------|-----------|---------|--------|--------|------|
| Llama    | 65B  | 128       | yes | 2,048 t              | 39,804 MB | 990 t/s | 20 t/s | 18 t/s | 4.20 |
| Llama    | 65B  | 32        | yes | 2,048 t              | 43,424 MB | 976 t/s | 17 t/s | 16 t/s | 4.11 |


### Testing long sequences

The following tests were all done on **30B/65B, 4bit 128g** with various settings, just to test the max sequence length
and get a sense of what can be achieved with different or multiple GPUs right now. Llama goes incoherent generating 
past 2048 tokens anyway, but with some fine-tuning, who knows? Note that these tests were run a while ago and the
speeds are no longer current.

|                        | Size | Seq. len. | VRAM                 | Long seq. | Ind.   | 
|------------------------|------|-----------|----------------------|-----------|--------|
| 4090/24GB              | 30B  | 2,516 t   | 22,145 MB            | 1140 t/s  | 28 t/s |
| 4090/24GB + 3070Ti/8GB | 30B  | 3,932 t   | 22,055 MB + 7,377 MB | 840 t/s   | 22 t/s |
| A6000/48GB (headless)  | 30B  | 9,032 t   | 46,863 MB            | 645 t/s   | 12 t/s |
| A100/80GB (headless)   | 65B  | 9,520 t   | 79,009 MB            | 650 t/s   | 9 t/s  |

## Todo

Moved the todo list [here](TODO.md).  

## Compatibility

I downloaded a whole bunch of GPTQ models to test compatibility. [Here](model_compatibility.md) is the list of models
confirmed to be working right now.

## Recent updates

**2023-05-22**: Added option to auto-split layers across multiple GPUs based on VRAM allocation. 

**2023-05-22**: Added option to dequantize layers at load-time which _should_ speed up inference, but it turns out
Torch's fp16 matmul is actually slower than the quantized matmul. Maybe bandwidth is the only bottleneck right now?
Need to experiment some more.

**2023-05-24**: Downloaded a bunch of models from HF and set up a test script. Should be a good sampling of the most
popular finetunes right now. I'll add more to the list as I come across them. They all seem to be working.

**2023-05-24**: Added fused rotary embeddings and some minor optimizations. 13% faster on 7B, 9% on 13B. Small
improvement on larger models. Added best-case scores to benchmark results and some clarification. For easier
comparisons to other implementations, or whatever.

**2023-05-27**: Better memory management in CUDA. Introduced auto switch between Torch's SDP backend and regular 
matmul attention with some tweaks. Finished CUDA MLP. All in all about 10% faster with these updates.

**2023-05-29**: Web UI is _almost_ up and running. Having to learn JavaScript, and it turns out I hate JavaScript. But
ChatGPT is an incredible resource for learning new languages, I gotta say, so it's not as painful as it could have
been. Anyway, in the process of working with the UI I discovered I've been measuring prompt speed incorrectly. Either
Torch or CUDA or the GPU driver does some sort of caching or self-calibration or lazy initialization during the first
pass through the model, so subsequent passes are actually _way_ faster than what I've been recording. Doesn't do much
for individual tokens, but benchmarks updated anyway. Closing in on 10k tokens/second for 7B. (!)

**2023-06-02**: Web UI is now in a fairly working state. Expect it to be a little scuffed in places. There will be a
rewrite at some point to make the client-side code less seizure-inducing. It has multibot mode, chat rewind and editing
features, sessions, and more. I'm going to build it out with support for instruct prompting and such, in time.

**2024-06-04**: Refactored a whole bunch to move more of the work into the extension, setting up for more tuning
options to come soon and eventually auto tuning. Also optimized a little, for about a 5% speedup.

**2024-06-06**: Some minor optimizations. Also it should now compile the extension more easily and run more seamlessly
on Windows.