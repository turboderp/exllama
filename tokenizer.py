from sentencepiece import SentencePieceProcessor
import os
import torch

class ExLlamaTokenizer:

    def __init__(self, tokenizer_model_path):

        self.path = tokenizer_model_path
        self.tokenizer = SentencePieceProcessor(model_file = self.path)

    def encode(self, text):

        ids = self.tokenizer.Encode(text)
        return torch.tensor(ids).unsqueeze(0)

    def decode(self, ids):

        ids = ids.tolist()
        text = self.tokenizer.Decode(ids)
        return text
