## Model compatibility

- [ ] Verify compatibility with Llama-2 34B once released

## GPU compatibility (etc.)

- [ ] Optimizations for ROCm
- [ ] Optimizations for RTX 20-series maybe
- [ ] Look into improving P40 performance

## Testing

- [ ] More testing on Llama 2 models

## Optimization

- [ ] Flash Attention 2.0 (?)
- [ ] Find a way to eliminate `ExLlamaAttention.repeat_kv` (custom attention kernel?)
- [ ] C++ implementations of sampler functions

## Generation

- [ ] Optimized/batched beam search
- [ ] Allow stackable LoRAs
- [ ] Guidance or equivalent

## Interface

- [ ] Comprehensive API server (more than `example_flask.py`

## Web UI

- [ ] Controls to enable beam search
- [ ] Rewrite/refactor all the JavaScript and CSS
- [ ] Make it a little prettier
- [ ] Better error handling
- [ ] LoRA controls
- [ ] Multiple chat modes with prompt templates (instruct, etc.)

## ??

- [ ] Support for other quantization methods
- [ ] Support for other LLM architectures
- [ ] Allow for backpropagation
- [ ] LoRA training features
- [ ] Soft prompt training