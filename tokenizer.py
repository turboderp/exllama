from sentencepiece import SentencePieceProcessor
import os
import torch

class ExLlamaTokenizer:

    def __init__(self, tokenizer_model_path):

        self.path = tokenizer_model_path
        self.tokenizer = SentencePieceProcessor(model_file = self.path)
        self.eos_token_id = self.tokenizer.eos_id()
        self.bos_token_id = self.tokenizer.bos_id()
        self.newline_token_id = 13

    def encode(self, text):

        ids = self.tokenizer.Encode(text)
        return torch.tensor(ids).unsqueeze(0)

    def decode(self, ids):

        ids = ids.tolist()
        text = self.tokenizer.Decode(ids)
        return text

    def num_tokens(self, text):

        ids = self.tokenizer.Encode(text)
        return len(ids)