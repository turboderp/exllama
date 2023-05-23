## Model compatibility

- [x] Support for act-order models ~~(a bit slow for now)~~
- [x] ~~Support for v1 models without groupsize~~ Nah.
- [ ] Test more models
- [ ] Consider support for loading GGML models
- [ ] Utility to scan and validate .safetensors files
- [ ] Figure out if there are quantized models with irregular groupsize (there are some at least with no groupsize)

## GPU compatibility

- [ ] Test that CUDA code works on GTX 10-series and RTX 20-series at some point
- [ ] Test performance on P40 (would be a good GPU to support)
- [ ] Tunable kernel parameters

## Testing

- [ ] Figure out an apples-to-apples way of comparing perplexity with other implementations
- [ ] Compile charts of inference speed vs context length for variety of models, compare to other implementations

## VRAM optimization

- [ ] Fix layer streaming so it isn't unusably slow
- [ ] Allow layer streaming to integrate with other features like device splitting
- [ ] Provide alternative backend to allow layers on CPU

## Speed optimization

- [x] Support for de-quantizing select matrices at load time
- [ ] Better vector-matrix multiplication for de-quantized matrices (or show that it's bandwidth-limited now)
- [ ] Fused QKV projection
- [ ] Fused MLP (done, still needs act-order support)
- [ ] Fused RoPE
- [ ] Build attention mask in CUDA rather than PyTorch
- [ ] Disable attention mask when it isn't needed
- [ ] Figure out why inference appears to be CPU-bound
- [ ] Measure PyTorch module overhead (nn.Modules aren't really needed for inference)
- [ ] Examine if scaled_dot_product_attention is actually the best attention method for single tokens
- [ ] Rewrite at least the quantized matmul kernel. Should be a bunch of special cases to consider  

## Generation

- [x] Memory-efficient beam search implementation
- [ ] Optimized beam search
- [ ] Multi-token censoring/de-censoring
- [ ] Multi-token repetition penalties
- [ ] (Multi) LoRA support
- [ ] Guided generation (chat with multiple bots at once, etc.)
- [ ] Multiple chat modes with prompt templates (instruct, etc.)

## Interface

- [ ] Simple web interface?
- [ ] API server 

## ??

- [ ] Allow for backpropagation
- [ ] LoRA training features