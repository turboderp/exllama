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
* `numpy` (only for reference implementation) 

## Results so far

Tests done on stock RTX 4090. Will be updated as implementation is optimized further. 

|                                         | Seq. len. | VRAM           | Long seq.     | Ind.       | Ppl      |
|-----------------------------------------|-----------|----------------|---------------|------------|----------|
| 7B 4bit 128g, HF                        | 2,048 t   | 8,940 MB       | 2,218 t/s     | 45 t/s     | 6.44     |
| 13B 4bit 128g, HF                       | 2,048 t   | 14,902 MB      | 1,413 t/s     | 36 t/s     | 5.61     |
| 30B 4bit 128g, HF <sup>2</sup>          | 256 t     | 21,063 MB      | 168 t/s       | 26 t/s     | 4.61     |
| 7B 4bit 128g, HF, 16-bit                | 2,048 t   | 6,058 MB       | 2,887 t/s     | 45 t/s     | 6.45     |
| 13B 4bit 128g, HF, 16-bit               | 2,048 t   | 10,563 MB      | 2,212 t/s     | 36 t/s     | 5.62     |
| 30B 4bit 128g, HF, 16-bit <sup>2</sup>  | 1,024 t   | 20,715 MB      | 850 t/s       | 23 t/s     | 4.60     |
| **7B 4bit 128g, ExLlama**               | 2,048 t   | **5,599 MB**   | **2,133 t/s** | **71 t/s** | **6.45** |
| **13B 4bit 128g, ExLlama**              | 2,048 t   | **9,759 MB**   | **1,536 t/s** | **50 t/s** | **5.62** |
| **30B 4bit 128g, ExLlama** <sup>1</sup> | 2,048 t   | **21,048 MB**  | **149 t/s**   | **14 t/s** | **4.60** |
| **30B 4bit 128g, ExLlama** <sup>2</sup> | 1,700 t   | **21,380 MB**  | **662 t/s**   | **26 t/s** | **4.60** |

<sup>1</sup> Quantized matmul is needed to allow full sequence length for now   
<sup>2</sup> Max sequence length reduced to avoid OoM

All results are inference over a longer sequence, with the last 128 tokens generated individually, up to the sequence
length specified. VRAM usage is as reported by PyTorch and does not include PyTorch's own overhead (CUDA kernels,
internal buffers etc.) This is somewhat unpredictable anyway. Best bet is to just optimize VRAM usage by the model,
probably aiming for 20 GB on a 24 GB GPU to ensure there is room for a desktop environment and all of Torch's
internals.

Perplexity is measured to verify the accuracy of the output. The dataset used is a small sample from WikiText, so
scores are not necessarily comparable to other Llama benchmarks.

## Todo

- [x] Rudimentary test harness
- [x] Compile small, representative dataset for testing (or grab a little bit of WikiText)
- [x] Make sure new model performs at least as well as reference model on perplexity
- [ ] Sort out dependencies and compatibility with latest GPTQ release
- [ ] Optimize memory usage in large matrix multiplications
- [ ] Consider Triton implementation
- [ ] Evaluate Triton implementation
- [ ] Write and test Triton implementation
- [ ] Test device mapping across multiple GPUs
- [ ] Provide alternative backend to allow layers on CPU
- [x] Eliminate need for HF tokenizer (use SentencePiece library directly)
- [ ] Library of basic sampling methods
- [ ] Memory-efficient beam search implementation
- [ ] (Multi) LoRA support for inference
- [ ] Allow for backpropagation
- [ ] LoRA training features