import torch
from model import ExLlama

class ExLlamaGenerator:

    class Settings:

        temperature = 0.80
        top_k = 20
        top_p = 0.65
        min_p = 0.02  # Do not consider tokens with probability less than this

        token_repetition_penalty_max = 1.2  # Repetition penalty for most recent tokens
        token_repetition_penalty_sustain = 50  # No. recent tokens to repeat penalty for
        token_repetition_penalty_decay = 50  # Gradually decrease penalty over this many tokens


    def __init__(self, model, tokenizer, cache):

        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.cache.current_seq_len = 0
        self.sequence = None
        self.settings = None


    def make_rep_mask(self, penalty_max, sustain, decay):

        rep_mask = torch.ones(self.model.config.vocab_size)

        v = penalty_max
        dv = (1.0 - penalty_max) / decay
        i = self.sequence.shape[-1] - 1
        beg = max(i - sustain - decay, -1)

        while i > beg:

            t = self.sequence[0, i]
            rep_mask[t] = torch.max(rep_mask[t], torch.tensor(v))

            sustain -= 1
            if sustain < 0: v += dv
            i -= 1

        return rep_mask


    def sample(self, logits, temperature, top_k, top_p, min_p):

        logits = logits[0, -1, :]

        logits /= temperature
        logits += 1e-8
        probs = torch.softmax(logits, dim = -1)

        # Top K

        top_probs, top_indices = torch.topk(probs, top_k)

        # Top P

        num_top_p_probs = 0
        cum_prob = top_probs[0].item()
        while True:
            num_top_p_probs += 1
            if num_top_p_probs == top_probs.shape[-1]: break
            if top_probs[num_top_p_probs].item() < min_p: break
            cum_prob += top_probs[num_top_p_probs].item()
            if cum_prob > top_p: break

        top_probs = top_probs[:num_top_p_probs]
        sampled_ind = torch.multinomial(torch.softmax(top_probs, dim = -1), 1)
        sampled_tokens = top_indices[sampled_ind]

        return sampled_tokens.unsqueeze(0)


    def gen_begin(self, in_tokens):

        self.sequence = in_tokens
        self.cache.current_seq_len = 0

        if in_tokens.shape[-1] >= 1:
            self.model(self.sequence[:, :-1], self.cache, preprocess_only = False)


    def gen_single_token(self):

        logits = self.model(self.sequence[:, -1:], self.cache)

        rep_mask = self.make_rep_mask(self.settings.token_repetition_penalty_max,
                                      self.settings.token_repetition_penalty_sustain,
                                      self.settings.token_repetition_penalty_decay)
        logits /= rep_mask
        token = self.sample(logits,
                            self.settings.temperature,
                            self.settings.top_k,
                            self.settings.top_p,
                            self.settings.min_p)

        return token


    def gen_accept_tokens(self, tokens):

        self.sequence = torch.cat((self.sequence, tokens), dim = 1)


    def generate_simple(self, prompt, settings = Settings(), max_new_tokens = 128):

        self.settings = settings

        ids = self.tokenizer.encode(prompt)
        self.gen_begin(ids)

        for i in range(max_new_tokens):
            token = self.gen_single_token()
            if token.item() == self.tokenizer.eos_token_id: break
            self.gen_accept_tokens(token)

        text = self.tokenizer.decode(self.sequence[0])
        return text
