from sentencepiece import SentencePieceProcessor
import os
import torch

class ExLlamaTokenizer:

    def __init__(self, tokenizer_model_path):

        self.path = tokenizer_model_path
        self.tokenizer = SentencePieceProcessor(model_file = self.path)

        self.unk_token = "<unk>"
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        self.unk_token_id = self.tokenizer.unk_id()
        self.eos_token_id = self.tokenizer.eos_id()
        self.bos_token_id = self.tokenizer.bos_id()
        self.pad_token_id = 0  # self.tokenizer.pad_id()
        self.newline_token_id = 13


    # Encode string

    def encode(self, text, return_mask = False, max_seq_len = 2048):

        if isinstance(text, list):

            # text is a list of strings

            list_ids = self.tokenizer.EncodeAsIds(text)
            max_length = max([len(ids) for ids in list_ids])

            needs_mask = False
            padded_ids = []
            for ids in list_ids:
                if len(ids) != len(list_ids[0]): needs_mask = True
                padding = torch.full((max_length - len(ids),), self.pad_token_id)
                sequence = torch.tensor(ids)
                padded_ids.append(torch.cat((padding, sequence), dim = 0).long())

            stacked_ids = torch.stack(padded_ids, dim = 0)

            if return_mask:
                if needs_mask:
                    mask_padding = torch.full((stacked_ids.shape[0], max_seq_len - stacked_ids.shape[1]), True, dtype = torch.bool, device = "cpu")
                    mask = stacked_ids != 0
                    mask = torch.cat((mask, mask_padding), dim = 1)
                    return stacked_ids, mask
                else:
                    return stacked_ids, None
            else:
                return stacked_ids

        else:

            # text is a single string

            ids = self.tokenizer.EncodeAsIds(text)
            stacked_ids = torch.tensor(ids).unsqueeze(0)

            if return_mask:
                return stacked_ids, None
            else:
                return stacked_ids

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