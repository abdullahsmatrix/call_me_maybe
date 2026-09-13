"""Decoding engine that implements a token-by-token generation loop.

It enforces grammar constraints in real-time so a small model cannot
hallucinate invalid outputs. We get logits from the model, ask the
grammar which tokens are allowed, mask invalid logits to -inf, and
force the model to pick only valid options.
"""


from typing import Union, Tuple, Any
import numpy as np
from src.grammar import TrieMatcher, NumberGrammar, IntegerGrammar, StringGrammar


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
    grammar: Union[TrieMatcher, NumberGrammar, IntegerGrammar, StringGrammar],
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

        if grammar.is_complete(accumulated_text):
            # Some states (e.g. mid-digits of a number) are both a legal
            # stopping point and a legal continuation - "1" is already a
            # complete JSON number, so is_complete() alone can't tell us
            # whether the model actually wants to stop here or keep going
            # with more digits. Check the model's own unconstrained top
            # choice: only stop if it would pick something outside the
            # grammar anyway (e.g. a comma or closing brace). Otherwise a
            # multi-digit number would always be cut off after one digit.
            raw_top = int(np.argmax(logits))
            if raw_top not in valid_tokens:
                break

        masked_logits = mask_logits(logits, valid_tokens)
        next_token_id = int(np.argmax(masked_logits))

        # id_to_token keys are strings (JSON round-trip forces string keys)
        accumulated_text += vocab['id_to_token'][str(next_token_id)]
        generated_token_ids.append(next_token_id)
        input_ids.append(next_token_id)

        # Cheap (no model call) look-ahead: if the value is now complete
        # and there is genuinely nothing left it could legally continue
        # with (e.g. a string's closing quote), stop right away instead
        # of spending an extra forward pass just to discover that next
        # iteration.
        if (
            grammar.is_complete(accumulated_text)
            and not grammar.get_valid_token_ids(accumulated_text)
        ):
            break

    return (accumulated_text, generated_token_ids)
