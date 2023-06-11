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
    def __init__(self, method="default", model=None, cache=None, tokenizer=None):
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


    def _next_logits(self, input_ids, last_id_only=True):
        n_logits = None
        a = 0
        while a < input_ids.shape[-1]:
            b = min(input_ids.shape[-1], a + 2048)
            n_logits = self.model.forward(input_ids[:, a:b], self.cache, last_id_only)
            a = b

        return n_logits


    def _tokenize(self, text):
        return self.tokenizer.encode(text)


    # This loads *and* tokenizes into chunks
    def load(self, dataset_path, context=2048, overlap=0):
        file_extension = os.path.splitext(dataset_path)[1]

        # JSON format
        if file_extension == '.jsonl' or file_extension == '.json':
            with open(dataset_path) as f:
                for line in f:
                    example = json.loads(line)["text"]
                    # FIX: this was the default behavior but may lead to unexpected results for someone using this blindly...
                    # Consider a min-length option?
                    if len(example) > 50: 
                        chunk = self._tokenize(example)
                        chunk = chunk[:, :context + 1]
                        self.dataset_chunks.append(chunk)
        # Raw Text
        else:
            with open(dataset_path) as f:
                text = f.read()

            tokens = self._tokenize(text)

            # overlap shouldn't be bigger than the context, also need at least one token for predicting last...
            if overlap >= context:
                overlap = context-2

            # We can't use torch.chunks since it want's to split things into equal sized chunks. Instead, let's do our own chunking
            start = 0
            while start < tokens.size(1):
                chunk = tokens[:, start:start+context]
                start += context - overlap
                self.dataset_chunks.append(chunk)


    def test(self, chunk_limit=sys.maxsize, tag=""):
        if not self.dataset_chunks:
            sys.exit(" xx ERROR: Empty dataset!")

        print(f" -- Testing {min(len(self.dataset_chunks), chunk_limit)} chunks ", end="")
        sys.stdout.flush()

        logprob_sum = 0.0
        logprob_count = 0

        chunk_count = 0

        for chunk in self.dataset_chunks:

            self._begin()

            input_ids = chunk[:, :-1]
            target_ids = chunk[:, 1:]

            logits = self._next_logits(input_ids, last_id_only=False)

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
