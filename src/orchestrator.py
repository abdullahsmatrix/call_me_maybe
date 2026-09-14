"""Orchestrate a constrained function-call generation flow.

This module drives the decoder and grammars to produce a single JSON
object containing the chosen function `name` and its `parameters`.
"""

import re
from typing import Optional

import numpy as np

from src.validation_models import FunctionCallResults, FunctionDef
from src.grammar import (
    TrieMatcher, NumberGrammar, IntegerGrammar, StringGrammar,
)
from src.decoder import generate_constrained
from llm_sdk import Small_LLM_Model

_QUOTED_OR_WORD_RE = re.compile(r"'[^']*'|\"[^\"]*\"|\S+")


def call_function(
    prompt: str,
    available_functions: list[FunctionDef],
    model: Small_LLM_Model,
    vocab: dict
) -> FunctionCallResults:

    instruction_text = _build_instruction_prefix(prompt, available_functions)
    context_ids = model.encode(instruction_text)[0].tolist()

    function_name, context_ids = _generate_function_name(
        available_functions,
        model,
        vocab,
        context_ids
    )

    # close the name field and open the argument object so the model
    # sees the JSON built so far
    scaffold_text = '", "arguments": {'
    scaffold_ids = model.encode(scaffold_text)[0].tolist()
    context_ids = context_ids + scaffold_ids

    function_def = _find_function_def(function_name, available_functions)
    parameters, context_ids = _generate_parameters(
        function_def,
        model,
        vocab,
        context_ids,
        prompt,
    )

    result = _build_result(prompt, function_name, parameters)
    return result


def _extract_prompt_candidates(prompt: str) -> list[str]:
    """Build the list of literal substrings a string parameter may copy.

    Every quoted phrase or bare word in the prompt is a candidate, plus
    everything after the prompt's last colon (covers "Keyword: rest of
    the request" style prompts, where the whole remainder is one value).

    Constraining string generation to substrings that are already
    exactly present in the request eliminates hallucination and
    repetition loops by construction: the model can only choose WHICH
    real span to copy, never invent new characters. This assumes the
    correct value is always a literal span of the prompt, which holds
    for every case seen so far but is a real assumption, not a
    guarantee - see StringGrammar fallback below when no candidates
    exist at all.
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        text = text.strip()
        if len(text) >= 2 and text[0] == text[-1] and text[0] in ('"', "'"):
            text = text[1:-1].strip()
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    colon_idx = prompt.rfind(':')
    if colon_idx != -1:
        add(prompt[colon_idx + 1:])

    for match in _QUOTED_OR_WORD_RE.findall(prompt):
        add(match.strip(".,!?;:"))

    return candidates


def _escape_for_display(text: str) -> str:
    """Escape backslashes/quotes so the prompt can be safely wrapped in
    our own "..." framing without colliding with quotes already inside
    it. A raw prompt containing a literal '"' (e.g. Say "hello" to
    {name}) would otherwise render as User: "Say "hello" to {name}" -
    a broken, self-nested quote structure right where the model needs
    to tell where the request ends and its own answer begins, which
    reliably derails function-name selection. This only affects what
    the model is shown; the real, unescaped prompt is still what gets
    returned in the result.
    """
    return text.replace('\\', '\\\\').replace('"', '\\"')


def _build_instruction_prefix(
    prompt: str, available_functions: list[FunctionDef]
) -> str:
    """Describe the available functions and frame the expected JSON answer."""
    lines = [
        "You are a function-calling assistant.",
        "Choose exactly one function that satisfies the user's request",
        "and provide its parameters.",
        "",
        "CRITICAL RULES FOR PARAMETER VALUES:",
        "- For number/integer parameters: extract the EXACT numbers the",
        "  user actually wrote in their request, never a placeholder or",
        "  example value.",
        "- For string parameters: extract the EXACT text being referred",
        "  to, copied verbatim from the request.",
        "",
        "How to read values out of the request (the function name and",
        "parameter names must always come from the list below, never",
        "from these illustrations):",
        "- \"add 91 grams using the fold method\" -> the number is 91,",
        "  the method text is \"fold\".",
        "- \"label the box 'kitchen items' as fragile\" -> the quoted",
        "  text is one value, \"fragile\" is the other value.",
        "",
        "Available functions:",
    ]
    for func in available_functions:
        param_items = [
            f"{name}: {p_type.type}"
            for name, p_type in func.parameters.items()
        ]
        param_list = ", ".join(param_items)
        func_call = (
            f"- {func.name}({param_list}): {func.description}"
        )
        lines.append(func_call)
    lines.append("")
    lines.append(f'User: "{_escape_for_display(prompt)}"')
    lines.append("")
    # The scaffold key is deliberately "function", not "name": a prompt
    # containing a template placeholder like {name} collides with a
    # {"name": " scaffold - the model reads it as the placeholder being
    # filled in and predicts a person's name instead of a function name.
    # These keys are only what the model is shown; the emitted JSON keys
    # come from FunctionCallResults.
    lines.append('Answer: {"function": "')
    return "\n".join(lines)


def _generate_function_name(
    available_functions: list[FunctionDef],
    model: Small_LLM_Model,
    vocab: dict,
    context_ids: list[int]
) -> tuple[str, list[int]]:
    """Use TrieMatcher grammar to generate valid function name.
    returns function name and the context extended with its generated tokens.
    """
    function_names: list = [func.name for func in available_functions]
    grammar = TrieMatcher(function_names, vocab)
    function_name_text, generated_ids = generate_constrained(
        model,
        context_ids,
        grammar,
        vocab
    )
    return function_name_text, context_ids + generated_ids


def _generate_parameters(
    function_def: FunctionDef,
    model: Small_LLM_Model,
    vocab: dict,
    context_ids: list[int],
    prompt: str,
) -> tuple[dict[str, float | str | int | bool], list[int]]:
    """Generate values for each parameter using appropriate grammar.

    Returns the parameters dict and the context extended with all
    generated tokens.
    """
    parameters: dict = {}
    prompt_candidates = _extract_prompt_candidates(prompt)
    used_candidates: set[str] = set()
    # per-string-slot candidate scores, kept so the slot -> candidate
    # assignment can be settled globally once every slot has bid
    score_table: dict[str, dict[str, float]] = {}

    for i, (param_name, param_type) in enumerate(
        function_def.parameters.items()
    ):
        # tell the model which parameter slot it is filling before
        # generating its value
        separator = '' if i == 0 else ', '
        label_ids = model.encode(f'{separator}"{param_name}": ')[0].tolist()
        context_ids = context_ids + label_ids

        if param_type.type == 'number':
            value_text, generated_ids = generate_constrained(
                model, context_ids, NumberGrammar(vocab), vocab
            )
            context_ids = context_ids + generated_ids
            parameters[param_name] = float(value_text)
        elif param_type.type == 'integer':
            value_text, generated_ids = generate_constrained(
                model, context_ids, IntegerGrammar(vocab), vocab
            )
            context_ids = context_ids + generated_ids
            parameters[param_name] = int(value_text)
        elif param_type.type == 'boolean':
            grammar = TrieMatcher(['true', 'false'], vocab)
            value_text, generated_ids = generate_constrained(
                model, context_ids, grammar, vocab
            )
            context_ids = context_ids + generated_ids
            parameters[param_name] = value_text == 'true'
        else:
            # Drop the parameter's own name: models readily echo the key
            # as its own value ("template": "template"), and a value
            # that is literally the parameter name is essentially never
            # what was asked for.
            candidates = [
                c for c in prompt_candidates
                if c.lower() != param_name.lower()
            ]

            # We write the JSON quotes ourselves since candidates are
            # plain (unquoted) text; score inside the opening quote so
            # the model sees the same context it would generate in.
            quote_ids = model.encode('"')[0].tolist()
            scores: dict[str, float] = {}
            token_ids_by_candidate: dict[str, list[int]] = {}
            if candidates:
                scores, token_ids_by_candidate = _score_string_candidates(
                    model, context_ids + quote_ids, candidates, quote_ids
                )

            chosen = _best_unused(scores, used_candidates)
            if chosen is not None:
                score_table[param_name] = scores
                value_text = chosen
                value_ids = token_ids_by_candidate[chosen]
                context_ids = context_ids + quote_ids + value_ids + quote_ids
            else:
                # degenerate prompt with nothing to copy - fall back to
                # free-form (but still grammar-constrained) generation,
                # which manages its own opening/closing quotes.
                string_grammar = StringGrammar(vocab)
                value_text, generated_ids = generate_constrained(
                    model, context_ids, string_grammar, vocab
                )
                context_ids = context_ids + generated_ids
                value_text = value_text.strip('"')

            used_candidates.add(value_text)
            parameters[param_name] = value_text

    parameters.update(_assign_candidates_globally(score_table))
    return parameters, context_ids


def _assign_candidates_globally(
    score_table: dict[str, dict[str, float]],
) -> dict[str, str]:
    """Settle slot -> candidate assignment across all string slots.

    Filling slots left to right lets an early slot take a candidate that
    a later slot needs far more: for "Read the file at /home/user/
    data.json with utf-8 encoding", "utf-8" narrowly outbids the path for
    the `path` slot, while `encoding` wants "utf-8" with near certainty -
    so greedy order yields the two values swapped. Assigning the most
    confident (slot, candidate) pairs first fixes that.

    Because string values are selected from a fixed candidate list rather
    than generated token by token, this can be settled after the fact;
    the only approximation is that each slot's scores were measured in a
    context threaded with the earlier provisional picks.
    """
    if len(score_table) < 2:
        return {}
    pairs = sorted(
        (
            (score, slot, candidate)
            for slot, scores in score_table.items()
            for candidate, score in scores.items()
        ),
        key=lambda pair: -pair[0],
    )
    assigned: dict[str, str] = {}
    taken: set[str] = set()
    for _, slot, candidate in pairs:
        if slot in assigned or candidate in taken:
            continue
        assigned[slot] = candidate
        taken.add(candidate)
    return assigned


def _score_string_candidates(
    model: Small_LLM_Model,
    context_ids: list[int],
    candidates: list[str],
    terminator_ids: list[int],
) -> tuple[dict[str, float], dict[str, list[int]]]:
    """Score how well each candidate substring fits this slot.

    Two things make the comparison fair across candidates of very
    different lengths:

    - MEAN log-probability per token, not total. Total (like greedy
      first-token choice) is biased towards short candidates, so a
      one-token echo of nearby context - "file", right after
      "fn_read_file" - beats a longer span that is far more likely taken
      as a whole.
    - Each candidate is scored WITH its closing quote, i.e. as the
      complete terminated JSON value rather than an unterminated prefix.
      Otherwise "Hello" scores well on its own merits even when the model
      plainly wants to continue with "Hello {user}'s profile!", because
      the cost of having to close the string right there is never
      counted.

    Returns the score per candidate and the token ids per candidate
    (without the terminator).
    """
    scores: dict[str, float] = {}
    token_ids_by_candidate: dict[str, list[int]] = {}
    for candidate in candidates:
        token_ids: list[int] = model.encode(candidate)[0].tolist()
        if not token_ids:
            continue
        scores[candidate] = _mean_token_logprob(
            model, context_ids, token_ids + terminator_ids
        )
        token_ids_by_candidate[candidate] = token_ids
    return scores, token_ids_by_candidate


def _best_unused(
    scores: dict[str, float], used: set[str]
) -> Optional[str]:
    """Highest-scoring candidate not already taken by an earlier slot."""
    available = {c: s for c, s in scores.items() if c not in used} or scores
    if not available:
        return None
    return max(available, key=lambda c: available[c])


def _mean_token_logprob(
    model: Small_LLM_Model,
    context_ids: list[int],
    token_ids: list[int],
) -> float:
    """Mean log-probability of `token_ids` continuing `context_ids`."""
    total = 0.0
    current = list(context_ids)
    for token_id in token_ids:
        logits = np.asarray(
            model.get_logits_from_input_ids(current), dtype=np.float64
        )
        shifted = logits - logits.max()
        total += float(shifted[token_id] - np.log(np.exp(shifted).sum()))
        current.append(token_id)
    return total / len(token_ids)


def _find_function_def(
    function_name: str,
    available_functions: list[FunctionDef]
) -> FunctionDef:
    for fun in available_functions:
        if fun.name == function_name:
            return fun
    raise ValueError(f"Function: {function_name} not found")


def _build_result(
    prompt: str,
    function_name: str,
    parameters: dict
) -> FunctionCallResults:
    return FunctionCallResults(
        prompt=prompt,
        name=function_name,
        parameters=parameters,
    )
