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
        self.unk_token_id = self.tokenizer.unk_id() # is the same as pad token id...
        self.eos_token_id = self.tokenizer.eos_id()
        self.bos_token_id = self.tokenizer.bos_id()
        self.pad_token_id = 0  # self.tokenizer.pad_id()
        self.newline_token_id = 13

        self.special_characters = [(self.bos_token, self.bos_token_id), (self.eos_token, self.eos_token_id), (self.unk_token, self.unk_token_id)] # for tokenzier encoding

    # Encode string

    def encode(self, text, return_mask = False, max_seq_len = 2048, add_bos = False, add_eos = False, encode_special_characters = False):

        if isinstance(text, list):

            # text is a list of strings

            list_ids = self.tokenizer.EncodeAsIds(text)

            # pad bos and eos

            if add_bos:
                for ids in list_ids: ids.insert(0, self.bos_token_id)
            if add_eos:
                for ids in list_ids: ids.append(self.eos_token_id)

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
            split_text = [text]

            # look for special characters
            if encode_special_characters:
                for special_character, special_token_id in self.special_characters:
                    temp_text = []
                    for segment in split_text:
                        if isinstance(segment, str) and special_character in segment:
                            # for each special character, append the text before the special character, then append the special character ID, then the rest of the text
                            parts = segment.split(special_character)
                            new_parts = []
                            for i, part in enumerate(parts):
                                new_parts.append(part)
                                if i < len(parts) - 1:  # add the special token id between parts, but not after the last part
                                    new_parts.append(special_token_id)
                            temp_text.extend(new_parts)
                        else:
                            temp_text.append(segment)
                    split_text = temp_text

            ids = []

            for text_chunk in split_text:
                if isinstance(text_chunk, str):
                    ids += self.tokenizer.EncodeAsIds(text_chunk)
                else:
                    ids.append(text_chunk)

            # pad bos and eos

            if add_bos:
              ids = [self.bos_token_id] + ids
            if add_eos:
              ids = ids + [self.eos_token_id]

            stacked_ids = torch.tensor(ids).unsqueeze(0)

            if return_mask:
                return stacked_ids, None
            else:
                return stacked_ids

    def decode(self, ids, decode_special_characters=False):
        
        special_ids = {id_: char for char, id_ in self.special_characters}  # create a lookup dictionary

        if ids.dim() > 1:
            
            texts = []
            for i in range(ids.shape[0]):
                seq = ids[i].tolist()
                seq = [t for t in seq if t != self.pad_token_id]

                if decode_special_characters:
                    text_parts = []
                    normal_ids = []  # list of lists
                    current_normal_ids = []  # current list of normal IDs
                    for idx, id_ in enumerate(seq):
                        if id_ in special_ids:
                            # Save the current list of normal IDs, then start a new one
                            normal_ids.append(current_normal_ids)
                            current_normal_ids = []
                            # Store special token as a string
                            text_parts.append(special_ids[id_])
                        else:
                            current_normal_ids.append(id_)
                    normal_ids.append(current_normal_ids)  # save the last segment of normal IDs
                    
                    decoded_segments = [self.tokenizer.Decode(segment) for segment in normal_ids]
                    for idx, decoded_segment in enumerate(decoded_segments):
                        text_parts.insert(2*idx, decoded_segment)
                    
                    texts.append("".join(text_parts))
                else:
                    if self.eos_token_id in seq:  # to not mess up special char decoding
                        seq = seq[:seq.index(self.eos_token_id)]
                    texts.append(self.tokenizer.Decode(seq))

            return texts

        else:
            
            ids = ids.tolist()

            if decode_special_characters:
                
                text_parts = []
                normal_ids = []  # list of lists
                current_normal_ids = []  # current list of normal IDs
                for idx, id_ in enumerate(ids):
                    if id_ in special_ids:
                        # Save the current list of normal IDs, then start a new one
                        normal_ids.append(current_normal_ids)
                        current_normal_ids = []
                        # Store special token as a string
                        text_parts.append(special_ids[id_])
                    else:
                        current_normal_ids.append(id_)
                normal_ids.append(current_normal_ids)  # save the last segment of normal IDs
                
                decoded_segments = [self.tokenizer.Decode(segment) for segment in normal_ids]
                for idx, decoded_segment in enumerate(decoded_segments):
                    text_parts.insert(2*idx, decoded_segment)
                
                text = "".join(text_parts)
            
            else:
              
                text = self.tokenizer.Decode(ids)

            return text


    def num_tokens(self, text, encode_special_characters = False):
        
        if encode_special_characters:
            
            ids = self.encode(text, encode_special_characters = True)
            return ids.size(1)
        
        else:
            
            ids = self.tokenizer.Encode(text)
            return len(ids)