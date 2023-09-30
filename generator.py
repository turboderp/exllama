import cuda_ext
from model import ExLlama, ExLlamaCache
from lora import ExLlamaLora
import torch
import torch.nn.functional as F

class ExLlamaGenerator:

    class Settings:

        temperature = 0.95
        top_k = 40                              # consider the most probable top_k samples, 0 to disable top_k sampling
        top_p = 0.65                            # consider tokens up to a cumulative probabiltiy of top_p, 0.0 to disable top_p sampling
        min_p = 0.0                             # Do not consider tokens with probability less than this
        typical = 0.0                           # Locally typical sampling threshold, 0.0 to disable typical sampling

        token_repetition_penalty_max = 1.15     # Repetition penalty for most recent tokens
        token_repetition_penalty_sustain = 256  # No. most recent tokens to repeat penalty for, -1 to apply to whole context
        token_repetition_penalty_decay = 128    # Gradually decrease penalty over this many tokens

        beams = 1
        beam_length = 1


    model: ExLlama
    sequence: torch.Tensor or None
    sequence_actual: torch.Tensor or None
    settings: Settings
    beams: int or None
    max_beam_length: int
    in_beam_search: True
    disallowed_tokens: list[int] or None
    lora: ExLlamaLora or None


    def __init__(self, model, tokenizer, cache):

        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.reset()


    def reset(self):

        self.cache.current_seq_len = 0
        self.sequence = None
        self.sequence_actual = None
        self.settings = ExLlamaGenerator.Settings()

        self.beams = None
        self.max_beam_length = 0
        self.in_beam_search = False
        self.disallowed_tokens = None
        self.lora = None


    def make_rep_mask(self, penalty_max, sustain, decay):

        return cuda_ext.ext_rep_penalty_mask_cpu(self.model.config.vocab_size, self.sequence, penalty_max, sustain, decay)


    def batched_sample(self, logits, temperature, top_k, top_p, min_p, typical, num = 1):

        if logits.shape[0] == 1: return self.sample(logits, temperature, top_k, top_p, min_p, typical, num)

        samples = []
        scores = []
        for i in range(logits.shape[0]):
            t, s = self.sample(logits[i, :, :], temperature, top_k, top_p, min_p, typical)
            samples.append(t)
            scores.append(s)

        return torch.cat(samples, dim = 0), torch.cat(scores, dim = 0)


    # Sample one token from logits with current settings

    def sample_current(self, logits, num = 1):

        return self.sample(logits,
                           self.settings.temperature,
                           self.settings.top_k,
                           self.settings.top_p,
                           self.settings.min_p,
                           self.settings.typical)


    # Sample one token from logits

    def sample(self, logits, temperature, top_k, top_p, min_p, typical, num = 1):

        # torch.manual_seed(42)

        if logits.dim() == 3: logits = logits[0, -1, :]
        elif logits.dim() == 2: logits = logits[-1, :]
        else: raise ValueError("Bad logits dimension")

        # Disallow tokens

        if self.disallowed_tokens is not None:
            logits[self.disallowed_tokens] = float("-inf")

        # Base probabilities

        logits /= temperature
        logits += 1e-8
        probs = torch.softmax(logits, dim = -1)

        # Top K

        if top_k == 0:
            top_probs, top_indices = torch.sort(probs, descending = True)
        else:
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = F.normalize(top_probs, p = 1, dim = -1)

        # Top P

        if top_p > 0.0:

            num_top_p_probs = 0
            cum_prob = top_probs[0].item()
            while True:
                num_top_p_probs += 1
                if num_top_p_probs == top_probs.shape[-1]: break
                if top_probs[num_top_p_probs].item() < min_p: break
                cum_prob += top_probs[num_top_p_probs].item()
                if cum_prob > top_p: break

            top_probs = top_probs[:num_top_p_probs]
            top_probs = F.normalize(top_probs, p = 1, dim = -1)
            top_indices = top_indices[:num_top_p_probs]

        # Locally typical sampling

        if typical > 0.0:

            epsilon = 1e-10
            log_probs = (top_probs + epsilon).log()
            neg_entropy = (top_probs * log_probs).sum()
            entropy_dev = (neg_entropy - log_probs).abs()
            _, entropy_dev_order = torch.sort(entropy_dev)

            top_probs = top_probs.gather(-1, entropy_dev_order)
            top_indices = top_indices.gather(-1, entropy_dev_order)

            num_typical_probs = 0
            cum_prob = top_probs[0].item()
            while True:
                num_typical_probs += 1
                if num_typical_probs == top_probs.shape[-1]: break
                cum_prob += top_probs[num_typical_probs].item()
                if cum_prob > typical: break

            top_probs = top_probs[:num_typical_probs]
            top_probs = F.normalize(top_probs, p = 1, dim = -1)
            top_indices = top_indices[:num_typical_probs]

        # Multinomial sampling from top_probs, kept in same order as top_indices

        sampled_ind = torch.multinomial(top_probs, top_probs.shape[-1] if num == -1 else min(num, top_probs.shape[-1]))
        sampled_tokens = top_indices[sampled_ind]
        sampled_probs = top_probs[sampled_ind]  # Return probs before second norm

        if sampled_tokens.shape[0] > 1:
            sampled_tokens, ind = sampled_tokens.sort()
            sampled_probs = sampled_probs[ind]

        return sampled_tokens.unsqueeze(0), sampled_probs.unsqueeze(0)


    def disallow_tokens(self, tokens):

        self.disallowed_tokens = tokens


    def gen_begin(self, in_tokens, mask = None):

        self.end_beam_search()

        self.sequence = in_tokens.clone()
        self.sequence_actual = in_tokens.clone()
        self.cache.current_seq_len = 0

        self.model.forward(self.sequence[:, :-1], self.cache, preprocess_only = True, lora = self.lora, input_mask = mask)


    def gen_begin_empty(self):

        self.end_beam_search()
        self.sequence = None
        self.sequence_actual = None
        self.cache.current_seq_len = 0


    def gen_begin_reuse(self, in_tokens, mask = None):

        self.end_beam_search()
        if self.sequence is None or self.cache.current_seq_len == 0:
            self.gen_begin(in_tokens, mask = mask)
            return 0

        # if in_tokens.shape[-1] < self.sequence.shape[-1]:
        #     self.sequence = self.sequence[:, :in_tokens.shape[-1]]

        reuse = 0
        while reuse < self.sequence.shape[-1] and reuse < in_tokens.shape[-1] and self.sequence[0, reuse] == in_tokens[0, reuse]:
            reuse += 1

        if reuse < 2:
            self.gen_begin(in_tokens, mask = mask)
            return 0

        # print (f"Reusing cache: {reuse} tokens")

        self.cache.current_seq_len = reuse - 1
        self.sequence = self.sequence[:, :reuse]
        self.sequence_actual = self.sequence.clone()

        if reuse < in_tokens.shape[-1]: self.gen_feed_tokens(in_tokens[:, reuse:], mask = mask)
        return reuse


    def gen_feed_tokens(self, in_tokens, mask = None):

        if self.sequence is None:
            self.gen_begin(in_tokens, mask = mask)
            return

        self.end_beam_search()

        start = self.sequence.shape[-1] - 1
        if start < 0:
            start = 0
            self.sequence = in_tokens.clone()
        else:
            self.sequence = torch.cat((self.sequence, in_tokens), dim = 1)

        if start < self.sequence.shape[-1] - 1:
            self.model.forward(self.sequence[:, start : -1], self.cache, preprocess_only = True, lora = self.lora, input_mask = mask)

        self.sequence_actual = self.sequence


    def gen_accept_token(self, token):

        self.end_beam_search()
        if self.sequence is None: self.sequence = token
        else: self.sequence = torch.cat((self.sequence, token), dim = 1)
        self.sequence_actual = self.sequence


    def gen_rewind(self, num_tokens):

        if num_tokens == 0: return
        self.end_beam_search()
        self.sequence = self.sequence[:, :-num_tokens]
        self.cache.current_seq_len -= num_tokens
        self.sequence_actual = self.sequence


    def gen_prune_right(self, tokens, mask = None):

        self.end_beam_search()
        if tokens > self.sequence.shape[-1] - 1: return
        self.gen_begin(self.sequence[:, tokens:], mask = mask)
        self.sequence_actual = self.sequence


    def gen_prune_to(self, min_tokens_to_keep, token_id, mask = None):

        self.end_beam_search()

        if self.gen_num_tokens() <= min_tokens_to_keep: return

        while self.gen_num_tokens() > min_tokens_to_keep:

            pruned = False
            for i in range(self.sequence.shape[-1] - 1):
                if self.sequence[0, i] == token_id:
                    self.sequence = self.sequence[:, i + 1:]
                    pruned = True
                    break

            if not pruned: return

        self.gen_begin(self.sequence, mask = mask)


    def gen_prune_left(self, num_tokens, mask = None):

        num_tokens = min(num_tokens, self.sequence_actual.shape[-1] - 1)

        if self.in_beam_search:
            self.end_beam_search()  # TODO: Try to avoid restarting beam search when generating past chunk boundary
            self.sequence = self.sequence[:, num_tokens:]
            self.begin_beam_search()
        else:
            self.sequence = self.sequence[:, num_tokens:]
            self.gen_begin(self.sequence, mask = mask)


    def gen_num_tokens(self):

        return self.sequence_actual.shape[-1]


    # Simple generator function

    def generate_simple(self, prompt, max_new_tokens = 128):

        self.end_beam_search()

        ids, mask = self.tokenizer.encode(prompt, return_mask = True, max_seq_len = self.model.config.max_seq_len)
        self.gen_begin(ids, mask = mask)

        max_new_tokens = min(max_new_tokens, self.model.config.max_seq_len - ids.shape[1])

        eos = torch.zeros((ids.shape[0],), dtype = torch.bool)
        for i in range(max_new_tokens):
            token = self.gen_single_token(mask = mask)
            for j in range(token.shape[0]):
                if token[j, 0].item() == self.tokenizer.eos_token_id: eos[j] = True
            if eos.all(): break

        text = self.tokenizer.decode(self.sequence[0] if self.sequence.shape[0] == 1 else self.sequence)
        return text


    # Apply repetition penalty with current  settings

    def apply_rep_penalty(self, logits):

        cuda_ext.ext_apply_rep_penalty_mask_cpu(self.sequence,
                                                self.settings.token_repetition_penalty_max,
                                                self.settings.token_repetition_penalty_sustain,
                                                self.settings.token_repetition_penalty_decay,
                                                logits)


    # Generate a single token with the current settings, append to sequence

    def gen_single_token(self, constraints = None, mask = None):

        self.end_beam_search()

        # Simple sampling case:

        if self.sequence is not None:

            logits = self.model.forward(self.sequence[:, -1:], self.cache, lora = self.lora, input_mask = mask)
            self.apply_rep_penalty(logits)

            logits[:, :, self.tokenizer.bos_token_id] = -10000.0

            if constraints is not None:

                for c in constraints: logits[:, :, c] += 10000.0
                logits[:, :, :] -= 10000.0

            token, _ = self.batched_sample(logits,
                                           self.settings.temperature,
                                           self.settings.top_k,
                                           self.settings.top_p,
                                           self.settings.min_p + 0.01 if constraints is not None else 0.0,
                                           self.settings.typical)

        else:

            # bos = torch.Tensor([[self.tokenizer.bos_token_id]]).long()
            # logits = self.model.forward(bos, self.cache)
            # self.cache.current_seq_len = 0

            if constraints is not None:
                token = constraints[0]
            else:
                token = torch.Tensor([[self.tokenizer.bos_token_id]]).long()

        self.gen_accept_token(token)
        return token


    # Beam search

    class Beam:

        sequence: torch.Tensor                  # tokens generated in beam
        probs: torch.Tensor                     # probability score per token
        cache: ExLlamaCache                     # cached keys/values for this beam
        current_seq_pos: int                    # position of beam in current sequence
        settings = None
        generator = None

        sampled_tokens: torch.Tensor
        sampled_probs: torch.Tensor
        moved: bool = False

        def __init__(self, settings, generator, first_token = None, first_prob = None, seq_pos = None):

            self.settings = settings
            self.generator = generator

            self.sequence = first_token.unsqueeze(0).unsqueeze(0) if first_token is not None else None
            self.probs = first_prob.unsqueeze(0).unsqueeze(0) if first_prob is not None else None
            self.cache = ExLlamaCache(self.generator.model, max_seq_len = self.settings.beam_length)

            self.current_seq_pos = seq_pos


        def __len__(self):

            return self.sequence.shape[-1]


        def clone(self):

            new = ExLlamaGenerator.Beam(self.settings, self.generator)
            new.sequence = self.sequence.clone()
            new.probs = self.probs.clone()
            new.cache = self.cache.clone()
            new.current_seq_pos = self.current_seq_pos
            new.sampled_tokens = self.sampled_tokens.clone()
            new.sampled_probs = self.sampled_probs.clone()
            new.moved = self.moved
            return new


        # List of references to this instance

        def advance(self):

            self.cache.roll_left()
            self.sequence = self.sequence[:, 1:]
            self.probs = self.probs[:, 1:]
            self.current_seq_pos += 1


        # Cumulative probabilities

        def cum_log_probs(self):

            cum_log_prob = torch.sum(torch.log(self.probs))
            return cum_log_prob

        def sampled_cum_log_probs(self):

            cum_log_prob = torch.sum(torch.log(self.probs))
            return torch.log(self.sampled_probs) + cum_log_prob


        # Insert current beam in sequence

        def to_sequence(self):

            # Extend generator sequence and cache if needed

            new_tokens = 0
            added_tokens = 0
            slen = self.generator.sequence.shape[-1]
            tlen = self.current_seq_pos + len(self)
            if tlen > slen:
                new_tokens = tlen - slen
                added_tokens = new_tokens
                self.generator.sequence = torch.cat((self.generator.sequence, self.sequence[:, -new_tokens:]), dim = 1)
                self.generator.cache.current_seq_len = tlen - 1

            # Determine how much of generator sequence needs to be updated

            new_tokens_ = new_tokens
            for i in range(new_tokens_, len(self)):
                if self.generator.sequence[0, -i - 1] != self.sequence[0, -i - 1]: new_tokens = i + 1

            # Update sequence and cache

            if new_tokens > added_tokens:
                self.generator.sequence[0, -new_tokens:] = self.sequence[0, -new_tokens:]

            if new_tokens > len(self) - 1: new_tokens = len(self) - 1
            if new_tokens > 0:
                self.cache.copy_states(self.generator.cache,
                                       len(self) - 1 - new_tokens, new_tokens,
                                       self.generator.cache.current_seq_len - new_tokens, new_tokens,
                                       0, 1, 0, 1)


        # Copy last column of cache to this beam (after generation)

        def record_last_cache_column(self):

            self.generator.cache.copy_states(self.cache,
                                             self.generator.cache.current_seq_len - 1, 1,
                                             len(self) - 1, 1,
                                             0, 1, 0, 1)


    def begin_beam_search(self):

        self.beams = None
        if self.settings.beams == 1 and self.settings.beam_length == 1: return

        self.in_beam_search = True
        # self.testl = []


    def beam_search(self):

        if self.settings.beams == 1 and self.settings.beam_length == 1: return self.gen_single_token()
        assert self.in_beam_search

        # Kludge: The first token returned with an empty context is generated without beam search
        if self.sequence is None: return self.gen_single_token()

        c_cache_len = self.cache.current_seq_len
        c_seq_len = self.sequence_actual.shape[-1]

        # Begin here

        max_beam_length = min(self.model.config.max_seq_len - self.settings.beam_length, self.settings.beam_length)
        while self.beams is None or len(self.beams[0]) < max_beam_length:

            if self.beams is None:

                # Initial tokens for initial beams

                # self.cache.debug()
                logits = self.model.forward(self.sequence[:, -1:], self.cache, lora = self.lora)

                cuda_ext.ext_apply_rep_penalty_mask_cpu(self.sequence,
                                                        self.settings.token_repetition_penalty_max,
                                                        self.settings.token_repetition_penalty_sustain,
                                                        self.settings.token_repetition_penalty_decay,
                                                        logits)

                tokens, probs = self.sample(logits,
                                            self.settings.temperature,
                                            self.settings.top_k,
                                            self.settings.top_p,
                                            self.settings.min_p,
                                            self.settings.typical,
                                            num = self.settings.beams)

                # self.cache is updated with k/v for last token
                # Setup initial beams

                self.beams = []
                while len(self.beams) < min(self.settings.beams, tokens.shape[-1]):

                    beam = ExLlamaGenerator.Beam(self.settings, self, tokens[0, len(self.beams)], probs[0, len(self.beams)], c_seq_len)
                    self.beams.append(beam)

            else:

                # Sample from each beam

                # print(len(self.beams), end = "")
                for beam in self.beams:

                    beam.to_sequence()

                    # self.cache.debug()
                    logits = self.model.forward(self.sequence[:, -1:], self.cache, lora = self.lora)

                    cuda_ext.ext_apply_rep_penalty_mask_cpu(self.sequence,
                                                            self.settings.token_repetition_penalty_max,
                                                            self.settings.token_repetition_penalty_sustain,
                                                            self.settings.token_repetition_penalty_decay,
                                                            logits)

                    tokens, probs = self.sample(logits,
                                                self.settings.temperature,
                                                self.settings.top_k,
                                                self.settings.top_p,
                                                self.settings.min_p,
                                                self.settings.typical,
                                                num = -1)

                    beam.sampled_tokens = tokens
                    beam.sampled_probs = probs

                    beam.record_last_cache_column()
                    self.cache.current_seq_len -= 1

                # Collect options for all beams

                tokens_ = []
                probs_ = []
                cum_log_probs_ = []
                beams_ = []
                for i, beam in enumerate(self.beams):
                    tokens_.append(beam.sampled_tokens.squeeze(0))
                    probs_.append(beam.sampled_probs.squeeze(0))
                    cum_log_probs_.append(beam.sampled_cum_log_probs().squeeze(0))
                    beams_.append(torch.Tensor([i] * beam.sampled_tokens.shape[-1]).to(torch.int))

                tokens_all = torch.cat(tokens_, dim = 0)
                probs_all = torch.cat(probs_, dim = 0)
                cum_log_probs_all = torch.cat(cum_log_probs_, dim = 0)
                beams_all = torch.cat(beams_, dim = 0)

                # Sort by cumulative probability

                cum_log_probs_all, ind = cum_log_probs_all.sort(descending = True)
                probs_all = probs_all[ind]
                tokens_all = tokens_all[ind]
                beams_all = beams_all[ind]

                # Reduce to beam limit

                cum_log_probs_all = cum_log_probs_all[:self.settings.beams]
                probs_all = probs_all[:self.settings.beams]
                tokens_all = tokens_all[:self.settings.beams]
                beams_all = beams_all[:self.settings.beams]

                # Re-sort by beam index

                beams_all, ind = beams_all.sort()
                cum_log_probs_all = cum_log_probs_all[ind]
                tokens_all = tokens_all[ind]
                probs_all = probs_all[ind]

                # test = [self.tokenizer.decode(beam.sequence) for beam in self.beams]

                # Rebuild beams/caches

                for beam in self.beams: beam.moved = False
                beams_new = []

                for i in range(len(beams_all)):

                    new_token = tokens_all[i]
                    new_prob = probs_all[i]
                    beam_idx = beams_all[i].item()

                    if not self.beams[beam_idx].moved:

                        self.beams[beam_idx].sequence = torch.cat((self.beams[beam_idx].sequence, new_token.unsqueeze(0).unsqueeze(0)), dim = 1)
                        self.beams[beam_idx].probs = torch.cat((self.beams[beam_idx].probs, new_prob.unsqueeze(0).unsqueeze(0)), dim = 1)
                        self.beams[beam_idx].moved = True
                        beams_new.append(self.beams[beam_idx])

                    else:

                        nbeam = self.beams[beam_idx].clone()
                        nbeam.sequence[:, -1] = new_token
                        nbeam.probs[:, -1] = new_prob
                        beams_new.append(nbeam)

                self.beams = beams_new


        # Beam length is filled up, select winning beam

        max_log_probs = float("-inf")
        best_beam = None
        best_beam_idx = -1
        for beam_idx, beam in enumerate(self.beams):
            beam_log_probs = beam.cum_log_probs()
            if beam_log_probs > max_log_probs:
                max_log_probs = beam_log_probs
                best_beam = beam
                best_beam_idx = beam_idx

        best_token = best_beam.sequence[:, 0]

        # Insert in sequence

        self.sequence[0, c_seq_len] = best_token
        self.sequence_actual = torch.cat((self.sequence_actual, best_token.unsqueeze(0)), dim = 1)

        # Copy cache state for winning beam

        best_beam.to_sequence()

        # Prune other beams that don't begin with the winning token

        beams_new = [best_beam]

        for idx, beam in enumerate(self.beams):
            if idx != best_beam_idx and beam.sequence[:, 0] == best_token:
                beams_new.append(beam)

        self.beams = beams_new

        # Advance all remaining beams and caches

        for beam in self.beams: beam.advance()

        # Done

        return best_token


    def end_beam_search(self):

        if not self.in_beam_search: return

        self.sequence = self.sequence_actual.clone()
        self.cache.current_seq_len = self.sequence.shape[-1] - 1
        self.in_beam_search = False


    def replace_last_token(self, token, seq = False):

        self.sequence_actual[:, -1] = token
        if seq: self.sequence[:, -1] = token


    def sequence_ends_with(self, tokens):

        if self.sequence_actual.shape[-1] < tokens.shape[-1] + 1: return False
        for i in range(tokens.shape[-1]):
            if self.sequence_actual[0, -i - 1] != tokens[0, -i - 1]: return False
        return True

