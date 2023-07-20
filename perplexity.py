from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator

import json
import math
import os
import sys
import torch
import torch.nn.functional as F

'''
Passing in model, cache, tokenizer is a total hack because we don't want to have to reinitialize (or move all the globals into a shared state model)
'''

class Perplexity:
    def __init__(self, method="default", model = None, cache = None, tokenizer = None):
        # This needs to be loaded by calling .load()
        self.dataset_chunks = []

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer

        self._begin()


    def _begin(self):
        if self.cache is None:
            self.cache = ExLlamaCache(self.model)
        else:
            self.cache.current_seq_len = 0


    def _next_logits(self, input_ids, apply_lora, last_id_only = True):
        # n_logits = []
        # a = 0
        # while a < input_ids.shape[-1]:
        #     b = min(input_ids.shape[-1], a + 2048)
        #     n_logits.append(self.model.forward(input_ids[:, a:b], self.cache, last_id_only, lora = apply_lora))
        #     a = b
        #
        # return torch.cat(n_logits, dim = 1)

        return self.model.forward(input_ids, self.cache, last_id_only, lora = apply_lora)


    def _tokenize(self, text):
        return self.tokenizer.encode(text)


    # Load raw dataset from a text file and tokenize into chunks. Each chunk can optionally truncated to allow for
    # evaluating the same data at different sequence lengths

    def load(self, dataset_path, chunk_size, chunk_truncate = None, overlap = 0, minlength = 0, json_key = "text"):

        file_extension = os.path.splitext(dataset_path)[1]

        # JSON format: Returned chunks may be of variable length, with each chunk representing one list item

        if file_extension == '.jsonl' or file_extension == '.json':
            with open(dataset_path) as f:
                for line in f:
                    example = json.loads(line)[json_key]
                    if len(example) > minlength:
                        chunk = self._tokenize(example)
                        chunk = chunk[:, :chunk_size]
                        if chunk_truncate is not None: chunk = chunk[:, :chunk_truncate]
                        self.dataset_chunks.append(chunk)

        # Raw Text: Returned chunks are fixed length windows of the entire tokenized dataset

        else:
            with open(dataset_path, encoding="utf-8") as f:
                text = f.read()

            tokens = self._tokenize(text)

            # overlap shouldn't be bigger than the context, also need at least one token for predicting last...
            if overlap >= chunk_size:
                overlap = chunk_size-2

            # We can't use torch.chunks since it want's to split things into equal sized chunks. Instead, let's do our own chunking
            start = 0
            while start < tokens.size(1):
                chunk = tokens[:, start:start + chunk_size]
                start += chunk_size - overlap
                if chunk_truncate is not None: chunk = chunk[:, :chunk_truncate]
                self.dataset_chunks.append(chunk)


    def test(self, chunk_limit = sys.maxsize, lora = None, tag = "", ppl_token = False):
        if not self.dataset_chunks:
            sys.exit(" xx ERROR: Empty dataset!")

        print(f" -- Testing {min(len(self.dataset_chunks), chunk_limit)} chunks", end="")
        sys.stdout.flush()

        logprob_sum = 0.0
        logprob_count = 0

        chunk_count = 0

        for chunk in self.dataset_chunks:

            self._begin()

            input_ids = chunk[:, :-1]
            target_ids = chunk[:, 1:]

            if ppl_token:
                logits_s = []
                for i in range(input_ids.shape[-1]):
                    logits_t = self._next_logits(input_ids[:, i : i + 1], lora, last_id_only = False)
                    logits_s.append(logits_t)
                logits = torch.cat(logits_s, dim = 1)
            else:
                logits = self._next_logits(input_ids, lora, last_id_only = False)

            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

            logprob_sum += token_log_probs.sum().item()
            logprob_count += target_ids.numel()

            if chunk_count % 10 == 0:
                print(".", end = "")
                sys.stdout.flush()

            chunk_count += 1
            if chunk_limit and chunk_count >= chunk_limit:
                break

        mean_log_prob = logprob_sum / logprob_count
        perplexity = math.exp(-mean_log_prob)

        print("")
        print(f" ** Perplexity{tag}: {perplexity:.4f}")


def add_args(parser):

    parser.add_argument("-ppl", "--perplexity", nargs = '?', const = 'default', metavar = "METHOD", help = "Perplexity benchmark. Optionally specify method: gptq-for-llama, llama.cpp (not yet implemented)")
    parser.add_argument("-ppl_ds", "--perplexity_dataset", metavar = "DATAPATH", type = str, help = "Load dataset for perplexity (JSONL if .jsonl, otherwise parses it as raw text)")
    parser.add_argument("-ppl_cn", "--perplexity_chunk_num", nargs = "?", type = int, help = "Number of chunks for perplexity benchmark", default = 100)
    parser.add_argument("-ppl_cs", "--perplexity_chunk_size", type = int, help = "Size of chunks for perplexity benchmark", default = 2048)
    parser.add_argument("-ppl_ct", "--perplexity_chunk_truncate", type = int, help = "Truncated size of chunks for perplexity benchmark", default = 2048)
    parser.add_argument("-ppl_co", "--perplexity_chunk_overlap", type = int, help = "Chunk overlap", default = 0)
    parser.add_argument("-ppl_cm", "--perplexity_chunk_min", type = int, help = "Minimum chunk length", default = 50)
    parser.add_argument("-ppl_key", "--perplexity_json_key", type = str, help = "Key to extract from JSON dataset, default: 'text'", default = "text")
    parser.add_argument("-ppl_t", "--perplexity_token", action = "store_true", help = "Run perplexity test on individual tokens, for debug purposes (slow)")


def post_parse(args):

    if not args.perplexity: return

    # GPTQ-for-LLaMa equivalent

    if args.perplexity == "gptq-for-llama":
        args.perplexity_dataset = "datasets/wikitext2.txt"
        args.perplexity_chunk_num = 128
        args.perplexity_chunk_size = 2048
        args.perplexity_chunk_truncate = 2048
        args.perplexity_chunk_overlap = 0
        args.perplexity_chunk_min = 0

    # Default dataset for legacy method

    if args.perplexity_dataset is None: args.perplexity_dataset = "datasets/wikitext2_val_sample.jsonl"

    print(f" -- Perplexity:")
    print(f" -- - Dataset: {args.perplexity_dataset}")
    print(f" -- - Chunks: {args.perplexity_chunk_num}")
    print(f" -- - Chunk size: {args.perplexity_chunk_size}" + (f" -> {args.perplexity_chunk_truncate}" if args.perplexity_chunk_truncate is not None else ""))
    print(f" -- - Chunk overlap: {args.perplexity_chunk_overlap}")
    print(f" -- - Min. chunk size: {args.perplexity_chunk_min}")
    print(f" -- - Key: {args.perplexity_json_key}")
    if args.perplexity_token: print("f -- - Per-token mode")

