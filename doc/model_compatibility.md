## Working models

As of **2023-05-24**, the following GPTQ models on HuggingFace all appear to be working:

- Neko-Institute-of-Science/LLaMA-7B-4bit-128g
- Neko-Institute-of-Science/LLaMA-13B-4bit-128g
- Neko-Institute-of-Science/LLaMA-30B-4bit-32g
- Neko-Institute-of-Science/LLaMA-30B-4bit-128g
- Neko-Institute-of-Science/LLaMA-65B-4bit-32g
- Neko-Institute-of-Science/LLaMA-65B-4bit-128g
- reeducator/bluemoonrp-13b
- reeducator/bluemoonrp-30b
- TehVenom/Metharme-13b-4bit-GPTQ
- TheBloke/airoboros-13B-GPTQ
- TheBloke/gpt4-x-vicuna-13B-GPTQ
- TheBloke/GPT4All-13B-snoozy-GPTQ
- TheBloke/guanaco-33B-GPTQ
- TheBloke/guanaco-65B-GPTQ
- TheBloke/h2ogpt-oasst1-512-30B-GPTQ <sup>1</sup> 
- TheBloke/koala-13B-GPTQ-4bit-128g
- TheBloke/Manticore-13B-GPTQ
- TheBloke/medalpaca-13B-GPTQ-4bit
- TheBloke/medalpaca-13B-GPTQ-4bit (compat version)
- TheBloke/Nous-Hermes-13B-GPTQ
- TheBloke/tulu-30B-GPTQ
- TheBloke/vicuna-13B-1.1-GPTQ-4bit-128g
- TheBloke/VicUnlocked-30B-LoRA-GPTQ
- TheBloke/wizard-mega-13B-GPTQ
- TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ
- TheBloke/Wizard-Vicuna-13B-Uncensored-GPTQ
- TheBloke/WizardLM-7B-uncensored-GPTQ
- TheBloke/WizardLM-30B-Uncensored-GPTQ
- Yhyu13/chimera-inst-chat-13b-gptq-4bit

<sup>1</sup> This particular model, uniquely, shows somewhat worse perplexity when matmul is done by the custom CUDA 
kernel rather than cuBLAS. Maybe it's extra sensitive to rounding errors for some reason? Either way, it does work.

## Non-working models

As of **2023-05-24**, I have found no models that don't work.

v1 models are still unsupported, as are pickle files.