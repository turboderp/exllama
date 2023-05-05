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

* `torch` tested on 2.0.0 with cu117
* `xformers` tested on 0.0.18 (or, comment out the xformers attention stuff which doesn't work anyway)
* `safetensors` 0.3.1
* `transformers` 4.28.0 (only for LlamaTokenizer, which will be removed soon)
* `gptq-llama` tested on 0.2, commit @eaa9955d8700dc8566f0c443054233e9c4503f66
* `sentencepiece`

These shouldn't be needed but PyCharm says they're referenced somewhere:

* `colorama`
* `numpy`

## Some preliminary results

Tests on stock RTX 4090:

|                             | Model     | Cache    | Inference | Total     | Max (actual) | Speed 1   | Speed 2  |
|-----------------------------|-----------|----------|-----------|-----------|--------------|-----------|----------|
| 7B 4bit 128g, HF            | 4,859 MB  | -        | 4,080 MB  | 8,940 MB  | 10,712 MB    | 2,342 t/s | 31 t/s   |
| 13B 4bit 128g, HF           | 8,393 MB  | -        | 6,509 MB  | 14,902 MB | 18,268 MB    | 1,267 t/s | 25 t/s   |
| 30B 4bit 128g, HF           | 19,071 MB | -        | OoM       | OoM       | OoM          | OoM       | OoM      |
|                             |           |          |           |           |              |           |          |
| 7B 4bit 128g, HF, 16-bit    | 3,796 MB  | -        | 2,252 MB  | 6,058 MB  | 7,670 MB     | 2,991 t/s | 31 t/s   |
| 13B 4bit 128g, HF, 16-bit   | 7,033 MB  | -        | 3,530 MB  | 10,563 MB | 12,370 MB    | 2,225 t/s | 25 t/s   |
| 30B 4bit 128g, HF, 16-bit * | 17,062 MB | -        | 3,689 MB  | 20,715 MB | 22,734 MB    | 996 t/s   | 17 t/s   |
|                             |           |          |           |           |              |           |          |
| 7B 4bit 128g, ExLlama       | 3,611 MB  | 1,023 MB | 966 MB    | 5,600 MB  | 7,062 MB     | 2,258 t/s | 66 t/s   |
| 13B 4bit 128g, ExLlama      | 6,827 MB  | 1,600 MB | 1,333 MB  | 9,760 MB  | 11,270 MB    | 1,498 t/s | 51 t/s   |
| 30B 4bit 128g, ExLlama **   | 17,036 MB | 3,119 MB | 893 MB    | 21,048 MB | 22,514 MB    | 150 t/s   | 14 t/s   |

*) 1024 tokens only. OoMs on full context length  
**) Only quantized matmul, hence the lower speed. Could run at full speed on a 24 GB GPU, but I'd have to close my
podcasts, so...

All results (except for #6) are for 1920-token sequence lengths (speed 1) grown to 2048 tokens one token at a time 
(speed 2). First six are the standard implementation in Transformers, loaded in 4-bit mode more or less following the
methods in [this repo](https://github.com/johnsmith0031/alpaca_lora_4bit). Last three use the new implementation.

* **Model** is the base VRAM usage of each model before any inference is run.
* **Cache** usage measures the size of the cache, which the new model pre-allocates in full to avoid concatenating the
cache on every inference step, which is very wasteful as it turns out.
* **Inference** is peak usage measured during inference. This is considerably higher than it should be right now, due to
the conversion of quantized parameters back into floats to use the much faster PyTorch matmul instead of the quant-cuda 
version, for large enough tensors. I'm hopeful this can be optimized a lot. Might look at fused matmul with Triton.
* **Total** sums up VRAM the model *should* be using, except for...
* **Max** is the actual VRAM usage as reported by `nvidia-smi`. Apparently PyTorch adds a bit of overhead for CUDA
kernels and whatnot. It seems very unpredictable, so maybe the behavior could be tweaked.

## Todo

- [ ] Rudimentary test harness
- [ ] Compile small, representative dataset for testing
- [ ] Make sure new model performs at least as well as reference model on perplexity 
- [ ] Optimize memory usage in large matrix multiplications
- [ ] Consider Triton implementation
- [ ] Evaluate Triton implementation
- [ ] Write and test Triton implementation
- [ ] Test device mapping across multiple GPUs
- [ ] Provide alternative backend to allow layers on CPU
- [ ] Eliminate need for HF tokenizer (use SentencePiece library directly)
- [ ] Library of basic sampling methods
- [ ] Memory-efficient beam search implementation
- [ ] (Multi) LoRA support for inference
- [ ] Allow for backpropagation
- [ ] LoRA training features