from sentencepiece import SentencePieceProcessor
import os
import torch

class ExLlamaTokenizer:

    def __init__(self, tokenizer_model_path):

        self.path = tokenizer_model_path
        self.tokenizer = SentencePieceProcessor(model_file = self.path)
        self.eos_token_id = self.tokenizer.eos_id()
        self.bos_token_id = self.tokenizer.bos_id()
        self.pad_token_id = 0
        self.newline_token_id = 13

    # Encode string

    def encode(self, text):

        if isinstance(text, list):

            # text is a list of strings

            list_ids = self.tokenizer.Encode(text)
            max_length = max([len(ids) for ids in list_ids])

            padded_ids = []
            for ids in list_ids:
                padding = torch.full((max_length - len(ids),), self.pad_token_id)
                sequence = torch.tensor(ids)
                padded_ids.append(torch.cat((padding, sequence), dim = 0))

            return torch.stack(padded_ids, dim = 0)

        else:

            # text is a single string

            ids = self.tokenizer.Encode(text)
            return torch.tensor(ids).unsqueeze(0)

    def decode(self, ids):

        if ids.dim() > 1:

            texts = []
            for i in range(ids.shape[0]):
                seq = ids[i].tolist()
                seq = [t for t in seq if t != self.pad_token_id]
                if self.eos_token_id in seq: seq = seq[:seq.index(self.eos_token_id)]
                texts.append(self.tokenizer.Decode(seq))
            return texts

        else:

            ids = ids.tolist()
            text = self.tokenizer.Decode(ids)
            return text

    def num_tokens(self, text):

        ids = self.tokenizer.Encode(text)
        return len(ids)