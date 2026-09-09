"""Decoding engine that implements a token-by-token generation loop.

It enforces grammar constraints in real-time so a small model cannot
hallucinate invalid outputs. We get logits from the model, ask the
grammar which tokens are allowed, mask invalid logits to -inf, and
force the model to pick only valid options.
"""


from typing import Union, Tuple, Any
import numpy as np
from src.grammar import TrieMatcher, NumberGrammar, StringGrammar


def mask_logits(logits: np.ndarray, valid_token_ids: list[int]) -> np.ndarray:
    """Mask invalid token logits to -inf, allowing only valid tokens"""
    masked = logits.copy()
    valid_set = set(valid_token_ids)

    for token_id in range(len(logits)):
        if token_id not in valid_set:
            masked[token_id] = -np.inf
    return masked


def generate_constrained(
    model: Any,
    input_ids: list[int],
    grammar: Union[TrieMatcher, NumberGrammar, StringGrammar],
    vocab: dict,
    max_iterations: int = 60,
) -> Tuple[str, list[int]]:
    """Generate tokens constrained by grammar rules.
    returns accumulated text and generated token ids.
    """
    accumulated_text: str = ""
    generated_token_ids: list = []
    input_ids = input_ids.copy()

    for i in range(max_iterations):
        valid_tokens: list = grammar.get_valid_token_ids(accumulated_text)
        if not valid_tokens:
            break  # grammar dead end, nothing valid can continue
        logits = model.get_logits_from_input_ids(input_ids)
        masked_logits = mask_logits(logits, valid_tokens)
        next_token_id = int(np.argmax(masked_logits))

        # id_to_token keys are strings (JSON round-trip forces string keys)
        accumulated_text += vocab['id_to_token'][str(next_token_id)]
        generated_token_ids.append(next_token_id)
        input_ids.append(next_token_id)
        if grammar.is_complete(accumulated_text):
            break

    return (accumulated_text, generated_token_ids)
