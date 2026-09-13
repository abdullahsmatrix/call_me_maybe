"""Orchestrate a constrained function-call generation flow.

This module drives the decoder and grammars to produce a single JSON
object containing the chosen function `name` and its `parameters`.
"""

import re

from src.validation_models import FunctionCallResults, FunctionDef
from src.grammar import TrieMatcher, NumberGrammar, IntegerGrammar, StringGrammar
from src.decoder import generate_constrained
from llm_sdk import Small_LLM_Model
from typing import Optional, Union

_SINGLE_QUOTED_RE = re.compile(r"'([^']*)'")
_DOUBLE_QUOTED_RE = re.compile(r'"([^"]*)"')


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

    # close the name field and open "parameters" so the model sees the
    # JSON built so far
    scaffold_text = '", "parameters": {'
    scaffold_ids = model.encode(scaffold_text)[0].tolist()
    context_ids = context_ids + scaffold_ids

    function_def = _find_function_def(function_name, available_functions)
    quoted_span = _find_unambiguous_quoted_span(prompt)
    parameters, context_ids = _generate_parameters(
        function_def,
        model,
        vocab,
        context_ids,
        quoted_span,
    )

    result = _build_result(prompt, function_name, parameters)
    return result


def _find_unambiguous_quoted_span(prompt: str) -> Optional[str]:
    """Return the text inside a quoted span in the prompt, but only when
    exactly one such span (single- or double-quoted) exists.

    Quoting a literal value is a generic natural-language convention, not
    tied to any specific function or keyword, so this generalizes across
    arbitrary prompts/function sets rather than pattern-matching one
    domain's phrasing. With zero or multiple quoted spans there is no
    unambiguous single answer, so no correction is offered.
    """
    spans: list[str] = (
        _SINGLE_QUOTED_RE.findall(prompt) + _DOUBLE_QUOTED_RE.findall(prompt)
    )
    if len(spans) == 1:
        return spans[0]
    return None


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
        "  to. Do NOT restate the parameter's own name inside the value",
        "  (write utf-8, not encoding=utf-8). Do NOT add any other prefix",
        "  either, and do NOT include surrounding quotes in the value.",
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
    lines.append(f'User: "{prompt}"')
    lines.append("")
    lines.append('Answer: {"name": "')
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
    quoted_span: Optional[str] = None,
) -> tuple[dict[str, float | str | int | bool], list[int]]:
    """Generate values for each parameter using appropriate grammar.

    Returns the parameters dict and the context extended with all
    generated tokens.
    """
    parameters: dict = {}
    quoted_span_used = False
    for i, (param_name, param_type) in enumerate(
        function_def.parameters.items()
    ):
        # tell the model which parameter slot it is filling before
        # generating its value
        separator = '' if i == 0 else ', '
        label_ids = model.encode(f'{separator}"{param_name}": ')[0].tolist()
        context_ids = context_ids + label_ids

        grammar: Union[NumberGrammar, IntegerGrammar, StringGrammar, TrieMatcher]
        if param_type.type == 'number':
            grammar = NumberGrammar(vocab)
        elif param_type.type == 'integer':
            grammar = IntegerGrammar(vocab)
        elif param_type.type == 'boolean':
            grammar = TrieMatcher(['true', 'false'], vocab)
        else:
            grammar = StringGrammar(vocab)
        # generate value with constraints
        value_text, generated_ids = generate_constrained(
            model,
            context_ids,
            grammar,
            vocab
        )
        context_ids = context_ids + generated_ids
        # parse based on type
        if param_type.type == 'number':
            # convert string to float
            parameters[param_name] = float(value_text)
        elif param_type.type == 'integer':
            parameters[param_name] = int(value_text)
        elif param_type.type == 'boolean':
            parameters[param_name] = value_text == 'true'
        else:
            # remove surrounding quotes: "\"hello\"" -> "hello"
            cleaned = value_text.strip('"')
            cleaned = _clean_string_value(cleaned, param_name)
            # If the prompt had exactly one unambiguous quoted literal and
            # the model's own generated text already contains it, prefer
            # the crisp original over whatever noise the model wrapped
            # around it - but only once per call, so it can't be forced
            # onto more than one parameter.
            if (
                quoted_span is not None
                and not quoted_span_used
                and quoted_span in cleaned
            ):
                cleaned = quoted_span
                quoted_span_used = True
            parameters[param_name] = cleaned
    return parameters, context_ids


def _clean_string_value(value: str, param_name: str) -> str:
    """Strip a leaked scaffold/label fragment from a generated string.

    The model sometimes prepends a fragment of our own prompt framing
    (e.g. "user:") or restates the parameter's own name ("encoding=")
    instead of emitting just the extracted value. These are the model's
    own artifacts, not part of the requested value, so they are stripped
    deterministically after generation rather than something the grammar
    itself should be asked to prevent.
    """
    cleaned = value.strip()
    for prefix in (f"{param_name}=", f"{param_name}:", "user:", "user_"):
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix):].lstrip()
            break
    return cleaned


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
