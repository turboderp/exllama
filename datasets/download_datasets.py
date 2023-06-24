# import torch
# from tokenizer import ExLlamaTokenizer
from datasets import load_dataset
import os

# Download samples from HF datasets to run equivalent GPTQ-for-LLaMa equivalent benchmark

def download_hf(filename, dataset, subset, split, key, div):

    print(f"Downloading from {dataset}: {subset}, split: {split} ...")
    hf_dataset = load_dataset(dataset, subset, split = split)
    data = div.join(hf_dataset[key])

    with open(filename, "w", encoding="utf-8") as f:
        f.write(data)

download_hf("wikitext2.txt", "wikitext", "wikitext-2-raw-v1", "test", "text", "\n\n")
download_hf("ptb.txt", "ptb_text_only", "penn_treebank", "validation", "sentence", "\n\n")
download_hf("ptb_new.txt", "ptb_text_only", "penn_treebank", "test", "sentence", " ")
