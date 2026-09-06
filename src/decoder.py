"""This module is the decoding engine. we implement token by token generation loop
that enforce grammar constraints in real-time. Why? because the model is too small
to be trusted and my hallucinate and generate ivalid outputs.
We get logits -> Ask grammar what tokens are allowed -> mask all invalid yo -inf.
The model is forced to pick valid options only.
"""


from typing import Union, Tuple
import numpy as np
from src.grammar import TrieMatcher, NumberGrammar, StringGrammar
from llm_sdk import Small_LLM_Model

def mask_logits(logits: np.ndarray, valid_token_ids: list[int]) -> np.ndarray:
    """Mask invalid token logits to -inf, allowing only valid tokens"""
    masked = logits.copy()
    valid_set = set(valid_token_ids)

    for token_id in range(len(logits)):
        if token_id not in valid_set:
            masked[token_id] = np.NINF
    return masked

def generate_constrained(
    model,
    input_ids: list[int],
    grammar: Union[TrieMatcher, NumberGrammar, StringGrammar],
    vocab: dict,
    max_iterations: int = 1000
) -> Tuple[str, list[int]]:
    """Generate tokens constrained by grammar rules.
    returns accumulated text and generated token ids.
    """
    accumulated_text: str = ""
    generated_token_ids: list = []
    
    for i in range(max_iterations):
        valid_tokens: list = grammar.get_valid_token_ids(accumulated_text)
        logits = model.get_logits_from_input_ids(input_ids)
        masked_logits = mask_logits(logits, valid_tokens)
        next_token_id = np.argmax(masked_logits)

        accumulated_text += vocab['id_to_token'][next_token_id]
        generated_token_ids.append(next_token_id)
        input_ids.append(next_token_id)
        if grammar.is_complete(accumulated_text):
            break
    
    return (accumulated_text, generated_token_ids)


