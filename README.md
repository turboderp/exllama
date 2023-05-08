# ExLlama

A rewrite of the HF transformers implementation of Llama with the following goals, among others:

* Designed for use with quantized weights
* Memory-efficient inference (not just attention)
* Mapping across multiple devices
* Built-in (multi) LoRA support
* Companion library of funky sampling functions

Disclaimer: This is currently a preview of a work in progress. Or maybe a proof of concept. Either way it will all
change, a lot. More to be added, much will be removed, etc. Don't use this yet.

## Dependencies

This list might be incomplete:

* `torch` tested on 2.1.0 (nightly) with cu118
* `safetensors` 0.3.1
* `sentencepiece`
* `ninja`

## Results so far

|                                         | Seq. len. | VRAM          | Long seq.     | Ind.       | Ppl      |
|-----------------------------------------|-----------|---------------|---------------|------------|----------|
| 7B 4bit 128g, HF                        | 2,048 t   | 8,940 MB      | 2,218 t/s     | 45 t/s     | 6.44     |
| 13B 4bit 128g, HF                       | 2,048 t   | 14,902 MB     | 1,413 t/s     | 36 t/s     | 5.61     |
| 30B 4bit 128g, HF <sup>1</sup>          | 256 t     | 21,063 MB     | 168 t/s       | 26 t/s     | 4.61     |
| 7B 4bit 128g, HF, 16-bit                | 2,048 t   | 6,058 MB      | 2,887 t/s     | 45 t/s     | 6.45     |
| 13B 4bit 128g, HF, 16-bit               | 2,048 t   | 10,563 MB     | 2,212 t/s     | 36 t/s     | 5.62     |
| 30B 4bit 128g, HF, 16-bit <sup>1</sup>  | 1,024 t   | 20,715 MB     | 850 t/s       | 23 t/s     | 4.60     |

<sup>1</sup> Max sequence length reduced to avoid OoM

Tests above are on a reference implementation based on Sterlind's repo
[here](https://github.com/sterlind/GPTQ-for-LLaMa/tree/eaa9955d8700dc8566f0c443054233e9c4503f66). They will not be
updated, to avoid having to maintain superfluous code. Here are runs on the new implementation. which will be updated:

|                                     | Seq. len. | VRAM      | Long seq. | Ind.   | Ppl  |
|-------------------------------------|-----------|-----------|-----------|--------|------|
| 7B 4bit 128g, ExLlama               | 2,048 t   | 5,096 MB  | 2,580 t/s | 81 t/s | 6.45 |
| 13B 4bit 128g, ExLlama              | 2,048 t   | 8,982 MB  | 1,938 t/s | 50 t/s | 5.62 |
| 30B 4bit 128g, ExLlama              | 2,048 t   | 20,558 MB | 1,142 t/s | 21 t/s | 4.60 |

All tests done on stock RTX 4090, running with a desktop environment, with a few other apps also using VRAM.

All results are inference over a longer sequence, with the last 128 tokens generated individually, up to the sequence
length specified. VRAM usage is as reported by PyTorch and does not include PyTorch's own overhead (CUDA kernels,
internal buffers etc.) This is somewhat unpredictable anyway. Best bet is to just optimize VRAM usage by the model,
probably aiming for 20 GB on a 24 GB GPU to ensure there is room for a desktop environment and all of Torch's
internals.

Perplexity is measured only to verify the accuracy of the output. The dataset used is a small sample from WikiText, and
scores are not necessarily comparable to other Llama benchmarks.

The following tests were all done on 30B 4bit 128g with various settings, just to test the max sequence length and get
a sense of what can be achieved with multiple GPUs right now. Llama goes incoherent generating past 2048 tokens anyway,
but with some fine-tuning, who knows? 

|                        | Seq. len.  | VRAM                 | Long seq. | Ind.   | Ppl  |
|------------------------|------------|----------------------|-----------|--------|------|
| 4090/24GB              | 2,454 t    | 22,145 MB            | 648 t/s   | 28 t/s | 4.60 |
| 4090/24GB + 3070Ti/8GB | 3,932 t    | 22,055 MB + 7,377 MB | 467 t/s   | 22 t/s | 4.60 |


## Todo

- [x] Rudimentary test harness
- [x] Compile small, representative dataset for testing (or grab a little bit of WikiText)
- [x] Make sure new model performs at least as well as reference model on perplexity
- [x] ~~Sort out dependencies and compatibility with latest GPTQ release~~
- [x] Integrate quant-cuda from Sterlind's old commit, remove dependency
- [x] Optimize memory usage in large matrix multiplications
- [x] ~~Consider Triton implementation~~ (Triton implementations are all slow right now, it seems)
- [x] Test device mapping across multiple GPUs
- [ ] Provide alternative backend to allow layers on CPU
- [ ] Consider fused QKV projection and fused MLP
- [x] Eliminate need for HF tokenizer (use SentencePiece library directly)
- [x] Basic sampling methods
- [ ] Memory-efficient beam search implementation
- [ ] (Multi) LoRA support for inference
- [ ] Allow for backpropagation
- [ ] LoRA training features