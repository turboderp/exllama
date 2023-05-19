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

I am developing on an RTX 4090 and an RTX 3070-Ti. Both cards support the CUDA kernel, but there might be
incompatibilities with older cards. I have no way of testing that right now.

I have no idea if this works on Windows/WSL, but feel free to try and contribute/give feedback.

## Dependencies

This list might be incomplete:

* `torch` tested on 2.1.0 (nightly) with cu118, might work with older CUDA versions also
* `safetensors` 0.3.1
* `sentencepiece`
* `ninja`

## Limitations

As of currently (working on it):

- No support for v1 models without groupsize
- All the models I've tested are groupsize 128. Other groupsizes should work in theory, though
- I've encountered models with nonstandard layouts and datatypes (e.g. float32 embedding table). It'll take a while
to make sure all the possible permutations are supported.

## How to

There is no installer or package at the moment, but try this:

    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
    
    pip install safetensors sentencepiece ninja

    git clone https://github.com/turboderp/exllama
    cd exllama

    python test_benchmark_inference.py -t <path_to_tokenizer.model> -c <path_to_config.json> \ 
      -m <path_to_model.safetensors> -p -ppl

The CUDA extension is loaded at runtime so there's no need to install it separately. It will be compiled on the first
run and cached to `~/.cache/torch_extensions/` which could take a little while. If nothing happens at first, give it
a minute to compile.

Chatbot example:

    python test_chatbot.py -t <path_to_tokenizer.model> -c <path_to_config.json> \
      -m <path_to_model.safetensors> -un "Jeff" -p prompt_chatbort.txt

## Results so far

### New implementation
|                                     | Seq. len. | VRAM      | Long seq. | Ind.   | Ppl  |
|-------------------------------------|-----------|-----------|-----------|--------|------|
| 7B 4bit 128g                        | 2,048 t   | 5,092 MB  | 2,458 t/s | 95 t/s | 6.45 |
| 13B 4bit 128g                       | 2,048 t   | 8,975 MB  | 1,483 t/s | 62 t/s | 5.62 |
| 30B 4bit 128g                       | 2,048 t   | 20,544 MB | 1,174 t/s | 31 t/s | 4.60 |
| 30B 4bit 128g act-order             | 2,048 t   | 20,558 MB | 1,087 t/s | 30 t/s | 4.55 |
| 30B 4bit 32g act-order <sup>1</sup> | 1,650 t   | 21,437 MB | 908 t/s   | 29 t/s | 4.52 |

<sup>1</sup> Can not achieve full sequence length without OoM (yet)

All tests done on stock RTX 4090, running with a desktop environment, with a few other apps also using VRAM.

All results are inference over a longer sequence, with the last 128 tokens generated individually, up to the sequence
length specified. VRAM usage is as reported by PyTorch and does not include PyTorch's own overhead (CUDA kernels,
internal buffers etc.) This is somewhat unpredictable anyway. Best bet is to just optimize VRAM usage by the model,
probably aiming for 20 GB on a 24 GB GPU to ensure there is room for a desktop environment and all of Torch's
internals.

Perplexity is measured only to verify the accuracy of the output. The dataset used is a small sample from WikiText, and
scores are not necessarily comparable to other Llama benchmarks.

### Testing long sequences
The following tests were all done on **30B/65B, 4bit 128g** with various settings, just to test the max sequence length
and get a sense of what can be achieved with different or multiple GPUs right now. Llama goes incoherent generating 
past 2048 tokens anyway, but with some fine-tuning, who knows? 

|                        | Size | Seq. len. | VRAM                 | Long seq. | Ind.   | 
|------------------------|------|-----------|----------------------|-----------|--------|
| 4090/24GB              | 30B  | 2,516 t   | 22,145 MB            | 1140 t/s  | 28 t/s |
| 4090/24GB + 3070Ti/8GB | 30B  | 3,932 t   | 22,055 MB + 7,377 MB | 840 t/s   | 22 t/s |
| A6000/48GB (headless)  | 30B  | 9,032 t   | 46,863 MB            | 645 t/s   | 12 t/s |
| A100/80GB (headless)   | 65B  | 9,520 t   | 79,009 MB            | 650 t/s   | 9 t/s  |

### Comparisons

For reference, here are the best results I was able to achieve with GPTQ-for-LLaMa for the same task, using 
Sterlind's repo [here](https://github.com/sterlind/GPTQ-for-LLaMa/tree/eaa9955d8700dc8566f0c443054233e9c4503f66) which
appears to be the fastest. The new Triton branch is, as far as I can tell, slower, and the CUDA versions have gotten
slower as well over time.

|                                         | Seq. len. | VRAM          | Long seq.     | Ind.       | Ppl      |
|-----------------------------------------|-----------|---------------|---------------|------------|----------|
| 7B 4bit 128g, HF                        | 2,048 t   | 8,940 MB      | 2,218 t/s     | 45 t/s     | 6.44     |
| 13B 4bit 128g, HF                       | 2,048 t   | 14,902 MB     | 1,413 t/s     | 36 t/s     | 5.61     |
| 30B 4bit 128g, HF <sup>1</sup>          | 256 t     | 21,063 MB     | 168 t/s       | 26 t/s     | 4.61     |
| 7B 4bit 128g, HF, 16-bit                | 2,048 t   | 6,058 MB      | 2,887 t/s     | 45 t/s     | 6.45     |
| 13B 4bit 128g, HF, 16-bit               | 2,048 t   | 10,563 MB     | 2,212 t/s     | 36 t/s     | 5.62     |
| 30B 4bit 128g, HF, 16-bit <sup>1</sup>  | 1,024 t   | 20,715 MB     | 850 t/s       | 23 t/s     | 4.60     |

<sup>1</sup> Max sequence length reduced to avoid OoM

## Todo

- [x] Support for act-order models ~~(a bit slow for now)~~
- [x] ~~Support for v1 models without groupsize~~ Nah.
- [ ] Fix layer streaming 
- [ ] Tests on a wider range of models and variants
- [ ] Consider support for loading GGML models
- [ ] Figure out an apples-to-apples way of comparing perplexity with other implementations
- [ ] Options for trading off memory usage for more performance (e.g. float32 tensors)
- [ ] Provide alternative backend to allow layers on CPU
- [ ] Fused QKV projection and fused MLP
- [ ] Support for de-quantizing select matrices at load time
- [ ] A web interface maybe?
- [ ] Memory-efficient beam search implementation
- [ ] More sampling features
- [ ] (Multi) LoRA support for inference
- [ ] Allow for backpropagation
- [ ] LoRA training features

## Recent updates

**2023-05-16**: Decided to keep a short update log.

**2023-05-16**: Reworked the way act-order models are handled. Rows are now shuffled at load time so zeros and scales
can be scanned sequentially. Left-hand side of the matmul is shuffled column-wise accordingly. Performance cost for 
act-order is quite small now.

**2023-05-16**: Removed the need to specify groupsize.

**2023-05-17**: Tested 32g models (30B weights take up a bit too much space still to work on 24 GB of VRAM with full
context.) Added error handling to C++/CUDA parts. Cleaned up and simplified the CUDA code a lot, preparing for fused
layers.

**2023-05-18**: Added basic layer streaming. Experimental for now. Modern GPUs should be designed for concurrent data
transfer and execution, and there should be enough bandwidth to stream in every *n*th layer while the preceding *n*-1
layers are processing. It's still too slow to be useful for generation, though. Also doesn't work with multiple GPUs.