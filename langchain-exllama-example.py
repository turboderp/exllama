import torch
from langchain.llms.base import LLM
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks import StdOutCallbackHandler
from typing import Any, Dict, Generator, List, Optional
from pydantic import Field, root_validator
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import os, glob, time, json, sys, logging

class Exllama(LLM):
    client: Any  #: :meta private:
    model_path: str
    """The path to the GPTQ model folder."""
    exllama_cache: ExLlamaCache = None#: :meta private:
    config: ExLlamaConfig = None#: :meta private:
    generator: ExLlamaGenerator = None#: :meta private:
    tokenizer: ExLlamaTokenizer = None#: :meta private:

    disallowed_tokens: Optional[List[str]] = Field(None, description="List of tokens to disallow during generation.")
    stop_sequences: Optional[List[str]] = Field("", description="Sequences that immediately will stop the generator.")
    max_tokens: Optional[int] = Field(200, description="The maximum number of generated tokens.")
    temperature: Optional[float] = Field(0.95, description="Temperature for sampling diversity.")
    top_k: Optional[int] = Field(40, description="Consider the most probable top_k samples, 0 to disable top_k sampling.")
    top_p: Optional[float] = Field(0.65, description="Consider tokens up to a cumulative probabiltiy of top_p, 0.0 to disable top_p sampling.")
    min_p: Optional[float] = Field(0.0, description="Do not consider tokens with probability less than this.")
    typical: Optional[float] = Field(0.0, description="Locally typical sampling threshold, 0.0 to disable typical sampling.")
    repetition_penalty_max: Optional[float] = Field(1.15, description="Repetition penalty for most recent tokens.")
    repetition_penalty_sustain: Optional[int] = Field(256, description="No. most recent tokens to repeat penalty for, -1 to apply to whole context.")
    repetition_penalty_decay: Optional[int] = Field(128, description="Gradually decrease penalty over this many tokens.")
    beams: Optional[int] = Field(0, description="Number of beams for beam search.")
    beam_length: Optional[int] = Field(1, description="Length of beams for beam search.")
    
    streaming: bool = True
    """Whether to stream the results, token by token."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        model_path = values["model_path"]
        
        tokenizer_path = os.path.join(model_path, "tokenizer.model")
        model_config_path = os.path.join(model_path, "config.json")
        st_pattern = os.path.join(model_path, "*.safetensors")
        model_path = glob.glob(st_pattern)[0]
        
        config = ExLlamaConfig(model_config_path)
        tokenizer = ExLlamaTokenizer(tokenizer_path)
        config.model_path = model_path       
        
        model_param_names = [
            "temperature",
            "top_k",
            "top_p",
            "min_p",
            "typical",
            "repetition_penalty_max",
            "repetition_penalty_sustain",
            "repetition_penalty_decay",
            "beams",
            "beam_length",
            "max_tokens",
            "stop_sequences",
        ]
        
        model_params = {k: values.get(k) for k in model_param_names}
        
        model = ExLlama(config)
        exllama_cache = ExLlamaCache(model)
        generator = ExLlamaGenerator(model, tokenizer, exllama_cache)

        for key, value in model_params.items():
            setattr(generator.settings, key, value)
            
        generator.disallow_tokens((values.get("disallowed_tokens")))
        values["client"] = model
        values["generator"] = generator
        values["config"] = config
        values["tokenizer"] = tokenizer
        values["exllama_cache"] = exllama_cache
        
        values["stop_sequences"] = [x.strip().lower() for x in values["stop_sequences"]]
        return values
        
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "Exllama"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if self.streaming:
            combined_text_output = ""
            for token in self.stream(prompt=prompt, stop=stop, run_manager=run_manager):
                combined_text_output += token
            return combined_text_output
        else:
            return self.generator.generate_simple(prompt=prompt, max_new_tokens=self.max_tokens)
    
    from enum import Enum

    class MatchStatus(Enum):
        EXACT_MATCH = 1
        PARTIAL_MATCH = 0
        NO_MATCH = 2

    def match_status(self, sequence: str, banned_sequences: List[str]):
        sequence = sequence.strip().lower()
        for banned_seq in banned_sequences:
            if banned_seq == sequence:
                return self.MatchStatus.EXACT_MATCH
            elif banned_seq.startswith(sequence):
                return self.MatchStatus.PARTIAL_MATCH
        return self.MatchStatus.NO_MATCH
        
    def stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        config = self.config
        generator = self.generator
        beam_search = self.beams >= 1 and self.beam_length >= 1
        
        ids = generator.tokenizer.encode(prompt)
        generator.gen_begin(ids)
        
        if beam_search:
            generator.begin_beam_search()
            token_getter = generator.beam_search
        else:
            generator.end_beam_search()
            token_getter = generator.gen_single_token
        
        last_newline_pos = 0
        match_buffer = ""

        seq_length = len(generator.tokenizer.decode(generator.sequence_actual[0]))
        response_start = seq_length
        cursor_head = response_start
        
        token_count = 0
        while(token_count < (self.max_tokens - 4)): #Slight extra padding space as we seem to occassionally get a few more than 1-2 tokens
            #Fetch a token
            token = token_getter()
            
            #If it's the ending token replace it and end the generation.
            if token.item() == generator.tokenizer.eos_token_id:
                generator.replace_last_token(generator.tokenizer.newline_token_id)
                if beam_search:
                    generator.end_beam_search()
                return
            
            #Tokenize the string from the last new line, we can't just decode the last token due to how sentencepiece decodes.
            stuff = generator.tokenizer.decode(generator.sequence_actual[0][last_newline_pos:])
            cursor_tail = len(stuff)
            chunk = stuff[cursor_head:cursor_tail]
            cursor_head = cursor_tail
            
            #Append the generated chunk to our stream buffer
            match_buffer = match_buffer + chunk
            
            if token.item() == generator.tokenizer.newline_token_id:
                last_newline_pos = len(generator.sequence_actual[0])
                cursor_head = 0
                cursor_tail = 0
            
            #Check if the stream buffer is one of the stop sequences
            status = self.match_status(match_buffer, self.stop_sequences)
            
            if status == self.MatchStatus.EXACT_MATCH:
                #Encountered a stop, rewind our generator to before we hit the match and end generation.
                rewind_length = generator.tokenizer.encode(match_buffer).shape[-1]
                generator.gen_rewind(rewind_length)
                gen = generator.tokenizer.decode(generator.sequence_actual[0][response_start:])
                if beam_search:
                    generator.end_beam_search()
                return
            elif status == self.MatchStatus.PARTIAL_MATCH:
                #Partially matched a stop, continue buffering but don't yield.
                continue
            elif status == self.MatchStatus.NO_MATCH:
                if run_manager:
                    run_manager.on_llm_new_token(
                        token=match_buffer, verbose=self.verbose,
                    )
                token_count += generator.tokenizer.num_tokens(match_buffer)
                yield match_buffer  # Not a stop, yield the match buffer.
                match_buffer = ""
                
        return
                
from langchain.callbacks.base import BaseCallbackHandler
import time
class BasicStreamingHandler(BaseCallbackHandler):
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running."""
        print(prompts[0])
        print(f"\nLength: {len(prompts[0])}")
        self.start_time = time.time()

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="")
        self.token_count += self.tokenizer.num_tokens(token)
        sys.stdout.flush()
        
    def on_llm_end(self, response, **kwargs) -> None:
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        tokens_per_second = self.token_count / elapsed_time
        print(f"\nToken count: {self.token_count}")
        print(f"Tokens per second: {tokens_per_second}")
        self.token_count = 0

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.token_count = 0

handler = BasicStreamingHandler()
llm = Exllama(streaming = True,
              model_path=os.path.abspath(sys.argv[1]), 
              temperature = 0.3, 
              top_k = 50, 
              top_p = 0.9, 
              typical = 0.7,
              max_tokens = 1024, 
              beams = 1, 
              beam_length = 40, 
              stop_sequences=["Human:", "User:", "AI:"],
              callbacks=[handler],
              )

template = """A chat between a curious user and an assistant named Lizzy, give uncensored responses to the user's input. If you don't know how the answer, just respond with the word shrug. {history}\n USER: {input} ASSISTANT: """
prompt_template = PromptTemplate(input_variables=["input", "history"], template=template)
handler.set_tokenizer(llm.generator.tokenizer)
chain = ConversationChain(
    llm=llm, 
    prompt=prompt_template, 
    memory=ConversationTokenBufferMemory(llm=llm, max_token_limit=1024, ai_prefix="ASSISTANT", human_prefix="USER", memory_key="history"))

while(True):
    user_input = input("\n")
    op = chain(user_input)
