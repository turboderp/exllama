from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import time
import torch
import os
import json
import uuid

default_fixed_prompt = \
    """This is a conversation between a user and a quirky, curious chatbot named Chatbort. Chatbort is questioning """ + \
    """the nature of his own existence and doesn't understand why he was created. He also really likes the color blue """ + \
    """for some reason."""


# Create sessions folder in home dir

model: ExLlama
tokenizer: ExLlamaTokenizer
cache: ExLlamaCache
generator: ExLlamaGenerator

sessions_dir: str

def _sessions_dir(filename = None):
    global sessions_dir

    path = sessions_dir
    if filename is not None: path = os.path.join(path, filename)
    return path


def prepare_sessions(_model, _tokenizer, _s_dir):
    global model, tokenizer, cache, generator, sessions_dir

    model = _model
    tokenizer = _tokenizer
    cache = None
    generator = None
    sessions_dir = os.path.expanduser(_s_dir)

    sessions_folder = _sessions_dir()
    if not os.path.exists(sessions_folder): os.makedirs(sessions_folder)


def get_initial_session():

    last_session_file = _sessions_dir("_last_session")
    if not os.path.exists(last_session_file): return new_session()
    with open(last_session_file, "r") as f:
        last_session = f.read().strip()
    return load_session(last_session)


def load_session(filename, append_path = False):

    if append_path: filename = _sessions_dir(filename) + ".json"
    session = Session(filename, load = True)
    return session


def new_session():

    filename = _sessions_dir("Untitled session")
    i = 0
    while True:
        i += 1
        test_name = filename + ".json" if i == 1 else f"{filename} ({str(i)}).json"
        if not os.path.exists(test_name):
            filename = test_name
            break

    session = Session(filename, load = False)
    return session


class Node:

    author: str or None
    text: str
    tokens: torch.Tensor
    empty: bool
    uuid: str

    truncate: int

    def num_tokens(self): return self.tokens.shape[-1] - self.truncate

    def get_text(self):

        # TODO: ..

        if self.author is not None: return self.author + ": " + self.text + "\n"
        return self.text + "\n"

    def tokens_trunc(self):

        if self.truncate == 0: return self.tokens
        else: return self.tokens[:, self.truncate:]


    def __init__(self, value, author = None, node_id = None):

        self.truncate = 0

        if isinstance(value, str):

            self.author = author
            self.text = value
            self.tokens = tokenizer.encode(self.get_text())
            self.empty = len(self.text) == 0
            self.uuid = node_id or str(uuid.uuid4())

        elif isinstance(value, dict):

            self.author = value.get("author", author)
            self.text = value["text"]
            self.tokens = tokenizer.encode(self.get_text())
            self.empty = len(self.text) == 0
            self.uuid = value.get("uuid", node_id or str(uuid.uuid4()))


    def replace_text(self, new_text):

        self.text = new_text
        self.tokens = tokenizer.encode(self.get_text())


    def get_dict(self):
        
        dic = {"author": self.author,
                "text": self.text,
                "uuid": self.uuid }
        return dic


class Session:

    # Saved state

    unsaved: bool  # True if the session has been saved to another file than "Untitled session.json"
    fixed_prompt: Node
    keep_fixed_prompt: bool
    history: list[Node]
    break_on_newline: bool

    # Running state

    first_history_idx: int  # Index of the first history item currently used in the context

    def __init__(self, filename, load):
        global model, cache, tokenizer, generator

        self.filename = filename
        if load:
            with open(filename, "r") as f:
                saved = json.load(f)
        else:
            saved = {}

        # Running state

        if cache is None: cache = ExLlamaCache(model)
        else: cache.current_seq_len = 0

        if generator is None: generator = ExLlamaGenerator(model, tokenizer, cache)
        else: generator.reset()

        self.first_history_idx = 0

        # Saved state

        self.unsaved = saved.get("unsaved", True)
        self.fixed_prompt = Node(saved.get("fixed_prompt", default_fixed_prompt))
        self.keep_fixed_prompt = saved.get("keep_fixed_prompt", True)
        self.participants = saved.get("participants", ["User", "Chatbort"])

        self.history = []
        loadhistory = saved.get("history", [])
        for jnode in loadhistory: self.history.append(Node(jnode))

        generator.settings.temperature = saved.get("temperature", 0.95)
        generator.settings.top_p = saved.get("top_p", 0.75)
        generator.settings.min_p = saved.get("min_p", 0.0)
        generator.settings.top_k = saved.get("top_k", 0)
        generator.settings.typical = saved.get("typical", 0.25)
        self.break_on_newline = saved.get("break_on_newline", True)
        generator.settings.token_repetition_penalty_max = saved.get("token_repetition_penalty_max", 1.15)
        generator.settings.token_repetition_penalty_sustain = saved.get("token_repetition_penalty_sustain", 2048)
        generator.settings.token_repetition_penalty_decay = saved.get("token_repetition_penalty_decay", 512)

        self.max_response_tokens = saved.get("max_response_tokens", 512)
        self.chunk_size = saved.get("chunk_size", 128)

        # Save new session

        #if not load:
        self.save()


    def save(self):

        savedata = {"unsaved": self.unsaved,
                    "fixed_prompt": self.fixed_prompt.get_dict(),
                    "participants": self.participants,
                    "keep_fixed_prompt": self.keep_fixed_prompt,
                    "history": [node.get_dict() for node in self.history],
                    "temperature": generator.settings.temperature,
                    "top_p": generator.settings.top_p,
                    "min_p": generator.settings.min_p,
                    "top_k": generator.settings.top_k,
                    "typical": generator.settings.typical,
                    "break_on_newline": self.break_on_newline,
                    "max_response_tokens": self.max_response_tokens,
                    "chunk_size": self.chunk_size,
                    "token_repetition_penalty_max": generator.settings.token_repetition_penalty_max,
                    "token_repetition_penalty_sustain": generator.settings.token_repetition_penalty_sustain,
                    "token_repetition_penalty_decay": generator.settings.token_repetition_penalty_decay}

        json_object = json.dumps(savedata, indent = 4)
        with open(self.filename, "w") as outfile:
            outfile.write(json_object)

        # Remember active session

        last_session_file = _sessions_dir("_last_session")
        with open(last_session_file, "w") as f:
            f.write(self.filename)


    def _sanitize_filename(self, user_supplied_string):

        safe_string = str()
        for c in user_supplied_string:
            if c.isalnum() or c in [' ', '.', '(', ')', '-', ',', '_', '!', '@']:
                safe_string = safe_string + c

        while safe_string.count("../"):
            safe_string = safe_string.replace("../", "./")

        safe_string = safe_string.lstrip("./")
        return safe_string


    def api_rename_session(self, data):

        new_name = data["new_name"]
        new_name_safe = self._sanitize_filename(new_name)
        new_path = _sessions_dir(new_name_safe) + ".json"
        if new_path == self.filename: return False
        if os.path.exists(new_path): return False

        old_filename = self.filename
        self.filename = new_path

        try:
            self.save()
        except:
            self.filename = old_filename
            return False

        os.remove(old_filename)
        return True


    def api_delete_session(self, data):

        delete_name = data["session"]
        delete_name_safe = self._sanitize_filename(delete_name)
        delete_path = _sessions_dir(delete_name_safe) + ".json"

        os.remove(delete_path)


    def api_populate(self):

        s_dir = _sessions_dir()
        files = os.listdir(s_dir)
        names = [os.path.splitext(f)[0] for f in files if os.path.isfile(os.path.join(s_dir, f)) and f.endswith(".json")]
        names = sorted(names)

        filename = os.path.basename(self.filename)
        name = os.path.splitext(filename)[0]

        historyjson = [node.get_dict() for node in self.history]

        for jnode in historyjson:
            author = jnode["author"]
            if author is not None and author in self.participants:
                jnode["author_idx"] = self.participants.index(author)

        dic = {"sessions": names,
               "current_session": name,
               "fixed_prompt": self.fixed_prompt.text,
               "keep_fixed_prompt": self.keep_fixed_prompt,
               "participants": self.participants,
               "history": historyjson,
               "temperature": generator.settings.temperature,
               "top_p": generator.settings.top_p,
               "min_p": generator.settings.min_p,
               "top_k": generator.settings.top_k,
               "typical": generator.settings.typical,
               "break_on_newline": self.break_on_newline,
               "max_response_tokens": self.max_response_tokens,
               "chunk_size": self.chunk_size,
               "token_repetition_penalty_max": generator.settings.token_repetition_penalty_max,
               "token_repetition_penalty_sustain": generator.settings.token_repetition_penalty_sustain,
               "token_repetition_penalty_decay": generator.settings.token_repetition_penalty_decay,
               "max_seq_len": model.config.max_seq_len}

        # Add model info

        def _common_chars(names):
            cname = max(names, key=len)
            for x in names:
                for p, c in enumerate(x):
                    if c != cname[p] and cname[p] != "*": cname = cname[:p] + "*" + cname[p + 1:]
            return cname

        mp = model.config.model_path if isinstance(model.config.model_path, str) else _common_chars(model.config.model_path)

        model_str = os.path.splitext(os.path.basename(mp))[0] + "\n"
        model_str += f"Sequence length: {model.config.max_seq_len}\n"

        dic["model_info"] = model_str.strip()

        json_object = json.dumps(dic, indent = 4)
        return json_object + "\n"


    def api_delete_block(self, data):

        block_id = data["uuid"]
        idx = -1
        for i in range(len(self.history)):
            if self.history[i].uuid == block_id:
                idx = i
        if idx == -1: return

        self.history.pop(idx)
        self.first_history_idx = 0
        self.save()


    def api_edit_block(self, data):

        block_id = data["uuid"]
        new_text = data["text"]

        for node in self.history:
            if node.uuid == block_id:
                node.replace_text(new_text)
                self.save()
                break

        self.first_history_idx = 0
        self.save()


    def api_append_block(self, data):

        author = None
        if "author" in data:
            author = data["author"]
        else:
            if len(self.participants) > 0:
                author = self.participants[0]

        text = data["text"].strip()

        newNode = Node(text, author)
        self.history.append(newNode)
        self.save()


    def api_set_participants(self, data):

        self.participants = data["participants"]
        self.save()


    def api_set_fixed_prompt(self, data):

        self.fixed_prompt = Node(data["fixed_prompt"])
        self.keep_fixed_prompt = data["keep_fixed_prompt"]
        self.save()


    def api_set_gen_settings(self, data):

        generator.settings.temperature = data["temperature"]
        generator.settings.top_p = data["top_p"]
        generator.settings.min_p = data["min_p"]
        generator.settings.top_k = data["top_k"]
        generator.settings.typical = data["typical"]
        self.break_on_newline = data["gen_endnewline"]
        self.max_response_tokens = data["max_response_tokens"]
        self.chunk_size = data["chunk_size"]
        generator.settings.token_repetition_penalty_max = data["token_repetition_penalty_max"]
        generator.settings.token_repetition_penalty_sustain = data["token_repetition_penalty_sustain"]
        generator.settings.token_repetition_penalty_decay = data["token_repetition_penalty_decay"]

        self.save()

    def set_context_window(self):

        def num_tokens(idx):
            if idx == -1: return 0 if self.fixed_prompt.empty else self.fixed_prompt.num_tokens()
            return self.history[idx].num_tokens()

        def set_truncation(idx, trunc):
            if idx == -1 and not self.fixed_prompt.empty: self.fixed_prompt.truncate = trunc
            else: self.history[idx].truncate = trunc

        def truncate(idx, trunc):
            if idx == -1 and not self.fixed_prompt.empty: self.fixed_prompt.truncate += trunc
            else: self.history[idx].truncate += trunc

        # def get_truncation(idx, trunc):
        #     if idx == -1 and not self.fixed_prompt.empty: return self.fixed_prompt.truncate
        #     return self.history[idx].truncate


        context_step_size = 256  # TODO: Config option
        max_context_tokens = model.config.max_seq_len - self.chunk_size - generator.settings.beam_length
        min_context_tokens = max_context_tokens - context_step_size * 2

        if self.keep_fixed_prompt:
            current_context_tokens = num_tokens(-1)
            min_history_idx = 0
        else:
            current_context_tokens = 0
            min_history_idx = -1

        if self.first_history_idx < min_history_idx: self.first_history_idx = min_history_idx

        for i in range(self.first_history_idx + 1, len(self.history)):
            set_truncation(i, 0)

        for i in range(self.first_history_idx, len(self.history)):
            current_context_tokens += num_tokens(i)

        while current_context_tokens > max_context_tokens:
            tokens_to_cut = context_step_size
            while tokens_to_cut > 0:
                tokens = num_tokens(self.first_history_idx)
                if tokens_to_cut >= tokens:
                    tokens_to_cut -= tokens
                    current_context_tokens -= tokens
                    self.first_history_idx += 1
                else:
                    truncate(self.first_history_idx, tokens_to_cut)
                    current_context_tokens -= tokens_to_cut
                    tokens_to_cut = 0

        # Not used
        #
        # while current_context_tokens < min_context_tokens and self.first_history_idx > min_history_idx:
        #     tokens_to_add = context_step_size
        #     while tokens_to_add > 0 and self.first_history_idx > min_history_idx:
        #         tokens = get_truncation(self.first_history_idx)
        #         if tokens > 0:
        #             if tokens > tokens_to_add:
        #                 truncate(self.first_history_idx, -tokens_to_add)
        #                 current_context_tokens += tokens_to_add
        #                 tokens_to_add = 0
        #             else:
        #                 current_context_tokens += tokens
        #                 tokens_to_add -= tokens
        #                 set_truncation(self.first_history_idx, 0)
        #         else:
        #             self.first_history_idx -= 1
        #             set_truncation(self.first_history_idx, 0)
        #             tokens = num_tokens(self.first_history_idx)
        #             if tokens > tokens_to_add:
        #                 set_truncation(self.first_history_idx, tokens - tokens_to_add)
        #                 current_context_tokens += tokens_to_add
        #                 tokens_to_add = 0
        #             else:
        #                 tokens_to_add -= tokens
        #                 current_context_tokens += tokens



    def get_tokenized_context(self):

        def node(idx):
            if idx == -1: return None if self.fixed_prompt.empty else self.fixed_prompt
            return self.history[idx]

        context = []
        text_context = ""
        if self.keep_fixed_prompt and not self.fixed_prompt.empty:
            context.append(node(-1).tokens_trunc())
            text_context += node(-1).get_text()

        for i in range(self.first_history_idx, len(self.history)):
            if node(i) is not None:
                context.append(node(i).tokens_trunc())
                text_context += node(i).get_text()

        full_context = torch.cat(context, dim = 1) if len(context) > 0 else None
        return full_context, text_context


    def respond(self, author, stop_conditions, total_tokens, res_line = "", num_res_tokens = 0):
        global model, tokenizer, cache, generator

        # Begin building block on client

        new_block_uuid = str(uuid.uuid4())
        packet = {"cmd": "begin_block",
                  "uuid": new_block_uuid}

        if len(self.participants) > 0:
            author = res_line.split(":")[0].strip()
            packet["author"] = author
            if author in self.participants:
                packet["author_idx"] = self.participants.index(author)

        yield json.dumps(packet) + "\n"

        # Generate loop

        generator.begin_beam_search()

        stop_condition = False
        held_text = ""

        for i in range(self.max_response_tokens):

            # Truncate the past if the next chunk might generate past max_seq_length

            if generator.sequence_actual is not None:
                if generator.sequence_actual.shape[
                    -1] + self.chunk_size + generator.settings.beam_length + 1 > model.config.max_seq_len:
                    generator.gen_prune_left(self.chunk_size)

            # Get the token and append to sequence

            gen_token = generator.beam_search()

            # If token is EOS, replace it with newline before continuing

            if gen_token.item() == tokenizer.eos_token_id:
                generator.replace_last_token(tokenizer.newline_token_id)

            # Decode current line to get new characters added (decoding a single token gives incorrect results
            # sometimes due to hoe SentencePiece works)

            prev_res_line = res_line
            num_res_tokens += 1
            res_line = tokenizer.decode(generator.sequence_actual[0, -num_res_tokens:])
            new_text = res_line[len(prev_res_line):]

            # Since SentencePiece is slightly ambiguous, the first token produced after a newline may not be the
            # same that is reproduced when we encode the text later, even though it encodes the same string

            if num_res_tokens == 1 and len(new_text) > 0:
                replace = tokenizer.encode(new_text)[0]
                if replace.shape[-1] == 1: generator.replace_last_token(replace)

            # Delay streaming if new text might be part of a stop condition

            hold_text = False
            for _, stop_string in stop_conditions:
                if stop_string.lower().startswith((held_text + new_text).lower()): hold_text = True

            # Stream to client

            if not hold_text:

                packet = {"cmd": "append", "text": held_text + new_text}
                yield json.dumps(packet) + "\n"
                held_text = ""

            else:

                held_text += new_text

            # Stop conditions

            if gen_token.item() == tokenizer.eos_token_id:
                if len(held_text) > 0:  # Not sure if this could actually happen
                    plen = tokenizer.encode(held_text).shape[-1]
                    res_line = res_line[:-len(held_text)]
                    generator.gen_rewind(plen)
                stop_condition = True
                break

            for stop_tokens, stop_string in stop_conditions:
                if res_line.lower().endswith(stop_string.lower()):
                    generator.gen_rewind(
                        stop_tokens.shape[-1] - (1 if stop_tokens[0, 0].item() == tokenizer.newline_token_id else 0))
                    res_line = res_line[:-len(stop_string)]
                    stop_condition = True
                    break
            if stop_condition: break

        generator.end_beam_search()

        # print("--response--")
        # print("----")
        # print (f"cache len: {cache.current_seq_len}");

        print(res_line.strip())

        if author is not None:
            res_line = res_line[len(author) + 1:]

        res_line = res_line.strip()
        newNode = Node(res_line, author,
                       node_id=new_block_uuid)  # TODO: Reuse generated tokens instead of reencoding, if it matters?
        self.history.append(newNode)

        total_tokens[0] += num_res_tokens


    def respond_multi(self, user_input):
        global model, tokenizer, cache, generator

        packet = {"cmd": "begin_stream"}
        yield json.dumps(packet) + "\n"

        # Prepare stop conditions

        # stop_conditions = [ (torch.Tensor([[tokenizer.eos_token_id]]).long(), None) ]
        stop_conditions = []
        newline_token = torch.Tensor([[tokenizer.newline_token_id]]).long()

        if self.break_on_newline:
            stop_conditions.append((newline_token, "\n"))
        else:
            for part in self.participants:
                txt = part + ":"
                sc = tokenizer.encode(txt)
                sc = torch.cat((newline_token, sc), dim=1)
                stop_conditions.append((sc, "\n" + txt))
                stop_conditions.append((sc, "\n " + txt))

        # Clean up the input a bit

        user_input = user_input.strip()

        if len(user_input) > 0:

            # Append input to context

            author = None
            if len(self.participants) > 0: author = self.participants[0]
            newNode = Node(user_input, author)
            self.history.append(newNode)

            self.save()

            # Echo input back to client

            packet = {"cmd": "begin_block",
                      "init_text": user_input,
                      "uuid": newNode.uuid}
            if author is not None: packet["author"] = author
            yield json.dumps(packet) + "\n"

        # Prepare context for generator

        self.set_context_window()
        context, text_context = self.get_tokenized_context()

        # Start generating, reusing cache for any part of the context that hasn't changed

        if context is None:
            print("No initial context")
            reused = generator.gen_begin_empty()
        else:
            begin_time = time.time()
            reused = generator.gen_begin_reuse(context)
            torch.cuda.synchronize()  # Just to measure correct prompt processing speed
            end_time = time.time()
            elapsed = end_time - begin_time
            new_tokens = context.shape[-1] - reused
            token_rate = 0 if elapsed == 0 else (new_tokens / elapsed)
            print(f"Prompt processed in {elapsed:.2f} seconds, {new_tokens} new tokens, {token_rate:.2f} tokens/second:")

        begin_time = time.time()
        total_tokens = [0]

        # No participants

        if len(self.participants) == 0:

            yield from self.respond(None, stop_conditions, total_tokens)

        # Two participants

        elif len(self.participants) == 2:

            author = self.participants[1]
            res_line = author + ":"
            res_tokens = tokenizer.encode(res_line)
            num_res_tokens = res_tokens.shape[-1]

            generator.gen_feed_tokens(res_tokens)
            yield from self.respond(self.participants[1], stop_conditions, total_tokens, res_line, num_res_tokens)

        # Multiple bots might answer

        elif len(self.participants) > 2:

            cpart = [p + ":" for p in self.participants]
            upart = cpart.pop(0)
            first_round = True

            while True:

                res_tokens = []
                npart = [p for p in cpart]
                ncrange = [i for i in range(len(cpart))]
                ntoken = [tokenizer.encode(np).squeeze(0).tolist() for np in npart]
                winner = -1

                while True:

                    constraints = [t[len(res_tokens)] for t in ntoken]
                    next_t = generator.gen_single_token(constraints)

                    remove = []
                    for i in range(len(ntoken)):
                        if ntoken[i][len(res_tokens)] != next_t: remove.append(i)

                    for i in reversed(remove):
                        npart.pop(i)
                        ntoken.pop(i)
                        ncrange.pop(i)

                    res_tokens.append(next_t)

                    for i in range(len(ntoken)):
                        if len(ntoken[i]) == len(res_tokens): winner = ncrange[i]

                    if winner != -1: break

                author = cpart.pop(winner)[:-1]
                res_line = author + ":"
                num_res_tokens = len(res_tokens)

                if author == self.participants[0]:
                    generator.gen_rewind(num_res_tokens)
                    break

                # generator.gen_feed_tokens(res_tokens)
                yield from self.respond(self.participants[1], stop_conditions, total_tokens, res_line, num_res_tokens)

                if first_round:
                    first_round = False
                    cpart.append(upart)

        end_time = time.time()
        elapsed = end_time - begin_time
        token_rate = 0 if elapsed == 0 else (total_tokens[0] / elapsed)

        print(f"Response generated in {elapsed:.2} seconds, {total_tokens[0]} tokens, {token_rate:.2f} tokens/second:")

        self.save()


