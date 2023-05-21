import torch
from model import ExLlama, ExLlamaCache

class ExLlamaGenerator:

    class Settings:

        temperature = 0.95
        top_k = 20
        top_p = 0.65
        min_p = 0.02  # Do not consider tokens with probability less than this

        token_repetition_penalty_max = 1.15  # Repetition penalty for most recent tokens
        token_repetition_penalty_sustain = 256  # No. most recent tokens to repeat penalty for
        token_repetition_penalty_decay = 128  # Gradually decrease penalty over this many tokens

        beams = 1
        beam_length = 1


    def __init__(self, model, tokenizer, cache):

        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.cache.current_seq_len = 0
        self.sequence = None
        self.sequence_actual = None
        self.settings = None

        self.beams = None
        self.beams_probs = None
        self.beams_cache = None
        self.beams_cache_swap = None
        self.max_beam_length = 0
        self.in_beam_search = False


    def make_rep_mask(self, penalty_max, sustain, decay):

        rep_mask = torch.ones(self.model.config.vocab_size)

        v = penalty_max
        dv = (1.0 - penalty_max) / decay
        i = self.sequence_actual.shape[-1] - 1
        beg = max(i - sustain - decay, -1)

        while i > beg:

            t = self.sequence_actual[0, i].item()
            rep_mask[t] = torch.max(rep_mask[t], torch.tensor(v))

            sustain -= 1
            if sustain < 0: v += dv
            i -= 1

        return rep_mask


    def sample(self, logits, temperature, top_k, top_p, min_p, num = 1):

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
        norm_probs = top_probs / torch.sum(top_probs, dim = -1)  # Was extra softmax here (..?)
        sampled_ind = torch.multinomial(norm_probs, norm_probs.shape[-1] if num == -1 else min(num, norm_probs.shape[-1]))
        sampled_tokens = top_indices[sampled_ind]
        sampled_probs = top_probs[sampled_ind]  # Return probs before second norm

        if sampled_tokens.shape[0] > 1:
            sampled_tokens, ind = sampled_tokens.sort()
            sampled_probs = sampled_probs[ind]

        return sampled_tokens.unsqueeze(0), sampled_probs.unsqueeze(0)


    def gen_begin(self, in_tokens):

        self.end_beam_search()

        self.sequence = in_tokens
        self.sequence_actual = in_tokens
        self.cache.current_seq_len = 0

        if in_tokens.shape[-1] >= 1:
            self.model(self.sequence[:, :-1], self.cache, preprocess_only = True)


    def gen_feed_tokens(self, in_tokens):

        self.end_beam_search()

        start = self.sequence.shape[-1] - 1
        if start < 0:
            start = 0
            self.sequence = in_tokens
        else:
            self.sequence = torch.cat((self.sequence, in_tokens), dim = 1)
        self.model(self.sequence[:, start:-1], self.cache, preprocess_only = True)

        self.sequence_actual = self.sequence


    def gen_accept_token(self, token):

        self.end_beam_search()
        self.sequence = torch.cat((self.sequence, token), dim = 1)
        self.sequence_actual = self.sequence


    def gen_rewind(self, num_tokens):

        self.end_beam_search()
        self.sequence = self.sequence[:, :-num_tokens]
        self.cache.current_seq_len -= num_tokens
        self.sequence_actual = self.sequence


    def gen_prune_right(self, tokens):

        self.end_beam_search()
        if tokens > self.sequence.shape[-1] - 1: return
        self.gen_begin(self.sequence[:, tokens:])
        self.sequence_actual = self.sequence


    def gen_prune_to(self, max_tokens, token_id):

        self.end_beam_search()

        if self.gen_num_tokens() <= max_tokens: return

        while self.gen_num_tokens() > max_tokens:

            pruned = False
            for i in range(self.sequence.shape[-1] - 1):
                if self.sequence[0, i] == token_id:
                    self.sequence = self.sequence[:, i + 1:]
                    pruned = True
                    break

            if not pruned: return

        self.gen_begin(self.sequence)


    def gen_num_tokens(self):

        return self.sequence.shape[-1]


    # Generate some number of tokens and append to

    def generate_simple(self, prompt, settings = Settings(), max_new_tokens = 128):

        self.end_beam_search()

        self.settings = settings

        ids = self.tokenizer.encode(prompt)
        self.gen_begin(ids)

        for i in range(max_new_tokens):
            token = self.gen_single_token()
            if token.item() == self.tokenizer.eos_token_id: break
            self.gen_accept_token(token)

        text = self.tokenizer.decode(self.sequence[0])
        return text


    # Generate a single token with the current settings, append to sequence

    def gen_single_token(self):

        self.end_beam_search()

        # Simple sampling case:

        logits = self.model(self.sequence[:, -1:], self.cache)

        rep_mask = self.make_rep_mask(self.settings.token_repetition_penalty_max,
                                      self.settings.token_repetition_penalty_sustain,
                                      self.settings.token_repetition_penalty_decay)
        logits /= rep_mask
        token, _ = self.sample(logits,
                               self.settings.temperature,
                               self.settings.top_k,
                               self.settings.top_p,
                               self.settings.min_p)

        self.gen_accept_token(token)
        return token


    # Beam search

    def begin_beam_search(self):

        self.beams = None
        self.beams_probs = None
        if self.settings.beams == 1 and self.settings.beam_length == 1: return
        self.beams_cache = ExLlamaCache(self.model, max_seq_len = self.settings.beam_length, batch_size = self.settings.beams)
        self.beams_cache_swap = ExLlamaCache(self.model, max_seq_len = self.settings.beam_length, batch_size = self.settings.beams)
        self.in_beam_search = True

    def beam_search(self):

        if self.settings.beams == 1 and self.settings.beam_length == 1: return self.gen_single_token()
        assert self.in_beam_search

        c_cache_len = self.cache.current_seq_len
        c_seq_len = self.sequence.shape[-1]

        # Token-by-token repetition penalty mask

        rep_mask = self.make_rep_mask(self.settings.token_repetition_penalty_max,
                                      self.settings.token_repetition_penalty_sustain,
                                      self.settings.token_repetition_penalty_decay)

        # Begin here

        max_beam_length = min(self.model.config.max_seq_len - self.settings.beam_length, self.settings.beam_length)
        while self.beams is None or self.beams.shape[-1] < max_beam_length:

            if self.beams is None:

                # Start first beams

                logits = self.model(self.sequence[:, -1:], self.cache)
                logits /= rep_mask

                tokens, probs = self.sample(logits,
                                            self.settings.temperature,
                                            self.settings.top_k,
                                            self.settings.top_p,
                                            self.settings.min_p,
                                            num = self.settings.beams)

                self.beams = tokens.transpose(0, 1)
                self.beams_probs = probs.transpose(0, 1)
                # self.cache.copy_states(self.beams_cache, self.cache.current_seq_len - 1, 1, 0, 1, 0, 1, 0, self.beams.shape[0])
                # self.beams_cache.current_seq_len = 1
                # self.cache.current_seq_len -= 1

            else:

                # Sample from each beam

                # beams_cum_probs = torch.prod(self.beams_probs, dim = 1)

                tokens_ = []
                probs_ = []
                beams_row_idx_ = []

                for row in range(self.beams.shape[0]):

                    # Insert current beam into sequence and cache

                    new_token = self.beams[row, -1].unsqueeze(0).unsqueeze(0)
                    self.sequence = torch.cat((self.sequence, new_token), dim = 1)

                    distinct = 1
                    for c in range(1, self.beams.shape[1] + 1):
                        if self.beams[row, -c].item() != self.sequence[0, -c].item():
                            distinct = c

                    if distinct > 1:
                        self.sequence[0, -distinct:] = self.beams[row, -distinct:]
                        self.beams_cache.copy_states(self.cache,
                                                     self.beams_cache.current_seq_len - distinct + 1, distinct - 1,
                                                     self.cache.current_seq_len - distinct + 1, distinct - 1,
                                                     row, 1, 0, 1)

                    # Generate from where current beam diverges from cached sequence

                    new_tokens = self.sequence[:, -1:]
                    # self.cache.current_seq_len -= (distinct - 1)

                    logits = self.model(new_tokens, self.cache)
                    logits /= rep_mask

                    tokens, probs = self.sample(logits,
                                                self.settings.temperature,
                                                self.settings.top_k,
                                                self.settings.top_p,
                                                self.settings.min_p,
                                                num = -1)

                    beams_row_idx = torch.zeros_like(tokens)
                    beams_row_idx.fill_(row)

                    # Collect results for beam

                    # probs *= beams_cum_probs[row]

                    tokens_.append(tokens.squeeze(0))
                    probs_.append(probs.squeeze(0))
                    beams_row_idx_.append(beams_row_idx.squeeze(0))

                    # Save k/v cache for temporary token

                    self.cache.copy_states(self.beams_cache,
                                           self.cache.current_seq_len - 1, 1,
                                           self.beams_cache.current_seq_len, 1,
                                           0, 1, row, 1)

                    # Remove temporary token unless this is the last beam

                    if row < self.beams.shape[0] - 1:

                        self.sequence = self.sequence[:, :-1]
                        self.cache.current_seq_len -= 1

                # Advance beams cache

                self.beams_cache.current_seq_len += 1

                # Select best scores from all beams, up to max no. beams

                tokens_all = torch.cat(tokens_, dim = 0)
                probs_all = torch.cat(probs_, dim = 0)
                beams_row_idx = torch.cat(beams_row_idx_, dim = 0)

                probs_all, ind = probs_all.sort(descending = True)
                tokens_all = tokens_all[ind]
                beams_row_idx = beams_row_idx[ind]

                probs_all = probs_all[:self.settings.beams]
                tokens_all = tokens_all[:self.settings.beams]
                beams_row_idx = beams_row_idx[:self.settings.beams]

                # Re-sort by beam index to build cache faster and do less swapping on next iteration

                beams_row_idx, ind = beams_row_idx.sort()
                tokens_all = tokens_all[ind]
                probs_all = probs_all[ind]

                # Rebuild beams and caches

                beams_swap = []
                probs_swap = []
                beams_row_idx = beams_row_idx.tolist()
                i = 0

                while i < len(beams_row_idx):

                    row = i
                    row_a = beams_row_idx[row]
                    while i < len(beams_row_idx) and beams_row_idx[i] == row_a: i += 1
                    rows = i - row

                    for j in range(rows):

                        nrow = torch.cat((self.beams[row_a, :], tokens_all[row + j].unsqueeze(0)), dim = 0)
                        beams_swap.append(nrow)

                        prow = torch.cat((self.beams_probs[row_a, :], probs_all[row + j].unsqueeze(0)), dim = 0)
                        probs_swap.append(prow)

                    self.beams_cache.copy_states(self.beams_cache_swap,
                                                 0, self.beams_cache.current_seq_len,
                                                 0, self.beams_cache.current_seq_len,
                                                 row_a, 1,
                                                 row, rows)

                self.beams_cache_swap.current_seq_len = self.beams_cache.current_seq_len

                self.beams = torch.stack(beams_swap, dim = 0)
                self.beams_probs = torch.stack(probs_swap, dim = 0)

                test = self.tokenizer.decode(self.beams)  ##########

                temp = self.beams_cache
                self.beams_cache = self.beams_cache_swap
                self.beams_cache_swap = temp

        # Beam length is filled up, select winning next token

        beams_cum_probs = torch.sum(torch.log(self.beams_probs), dim = 1)
        winner = torch.argmax(beams_cum_probs, dim = 0)
        token = self.beams[winner, 0]

        self.sequence[0, c_seq_len] = token
        self.sequence_actual = torch.cat((self.sequence_actual, token.unsqueeze(0).unsqueeze(0)), dim = 1)

        # Copy cache state for winning token

        self.beams_cache.copy_states(self.cache,
                                     0, 1,
                                     c_cache_len, 1,
                                     winner, 1,
                                     0, 1)

        # Prune beams that don't begin with the winning token

        beams_swap = []
        probs_swap = []

        for i in range(self.beams.shape[0]):

            if self.beams[i, 0] != token: continue

            self.beams_cache.copy_states(self.beams_cache_swap,
                                         0, self.beams_cache.current_seq_len,
                                         0, self.beams_cache.current_seq_len,
                                         i, 1,
                                         len(beams_swap), 1)

            beams_swap.append(self.beams[i, :])
            probs_swap.append(self.beams_probs[i, :])

        self.beams_cache_swap.current_seq_len = self.beams_cache.current_seq_len

        self.beams = torch.stack(beams_swap, dim = 0)
        self.beams_probs = torch.stack(probs_swap, dim = 0)

        test = self.tokenizer.decode(self.beams)  ##########

        beams_cache_temp = self.beams_cache
        self.beams_cache = self.beams_cache_swap
        self.beams_cache_swap = beams_cache_temp

        # Remove first column of all beams

        self.beams_cache.roll_left()
        self.beams = self.beams[:, 1:]
        self.beams_probs = self.beams_probs[:, 1:]

        # Done

        return token


    def end_beam_search(self):

        if not self.in_beam_search: return

        self.sequence = self.sequence_actual
        self.cache.current_seq_len = self.sequence.shape[-1] - 1


    def replace_last_token(self, token):

        self.sequence_actual[:, -1] = token
        # self.sequence[:, self.sequence_actual.shape[-1] - 1] = token
        pass