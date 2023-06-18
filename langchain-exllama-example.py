import torch
from langchain.llms.base import LLM
from langchain.chains import ConversationChain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks import StdOutCallbackHandler
from typing import Any, Dict, Generator, List, Optional
from pydantic import Field, root_validator
from model import ExLlama, ExLlamaCache, ExLlamaConfig
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
        generator = ExLlamaGenerator(model, tokenizer, exllama_cache)   # create generator

        for key, value in model_params.items():
            setattr(generator.settings, key, value)
            
        generator.disallow_tokens((values.get("disallowed_tokens")))
        values["client"] = model
        values["generator"] = generator
        values["config"] = config
        values["tokenizer"] = tokenizer
        values["exllama_cache"] = exllama_cache
        
        values["stop_sequences"] = [x.strip() for x in values["stop_sequences"]]
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
        sequence = sequence.strip()
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

        response_start = len(generator.tokenizer.decode(generator.sequence_actual[0]))
        cursor_head = response_start
        for i in range(self.max_tokens):
            #Fetch a token
            token = token_getter()
            
            #If it's the ending token replace it and end the generation.
            if token.item() == generator.tokenizer.eos_token_id:
                generator.replace_last_token(generator.tokenizer.newline_token_id)
                if beam_search:
                    generator.end_beam_search()
                return
            
            #Tokenize the string from the last new line, we can't just decode the last token due to hwo sentencepiece decodes.
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
            
            #Check if th stream buffer is one of the stop sequences
            status = self.match_status(match_buffer, self.stop_sequences)
            
            if status == self.MatchStatus.EXACT_MATCH:
                #Encountered a stop, rewind our generator to before we hit the match and end generation.
                rewind_length = generator.tokenizer.encode(match_buffer).shape[-1]
                generator.gen_rewind(rewind_length)
                gen = generator.tokenizer.decode(generator.sequence_actual[0][response_start:])
                return gen
            elif status == self.MatchStatus.PARTIAL_MATCH:
                #Partially matched a stop, continue buffering but don't yield.
                continue
            elif status == self.MatchStatus.NO_MATCH:
                if run_manager:
                    run_manager.on_llm_new_token(
                        token=match_buffer, verbose=self.verbose,
                    )
                yield match_buffer  # Not a stop, yield the match buffer.
                match_buffer = ""
                
from langchain.callbacks.base import BaseCallbackHandler

class BasicStreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="")
        sys.stdout.flush()


llm = Exllama(streaming = True,
              model_path=os.path.abspath(sys.argv[1]), 
              temperature = 0.2, 
              top_k = 18, 
              top_p = 0.7, 
              max_tokens = 1000, 
              beams = 1, 
              beam_length = 40, 
              stop_sequences=["Human:", "User:", "AI:", "###"],
              callbacks=[BasicStreamingHandler()],
              )

chain = ConversationChain(llm=llm)
while(True):
    user_input = input("\n")
    prompt = f"""
    ### Instruction: 
    You are an extremely serious chatbot. Do exactly what is asked of you and absolutely nothing more.
    ### User:
    {user_input}
    ### Response:

    """
    op = chain(prompt)