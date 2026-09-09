"""Orchestrate a constrained function-call generation flow.

This module drives the decoder and grammars to produce a single JSON
object containing the chosen function `name` and its `parameters`.
"""

from src.validation_models import FunctionCallResults, FunctionDef
from src.grammar import TrieMatcher, NumberGrammar, StringGrammar
from src.decoder import generate_constrained
from llm_sdk import Small_LLM_Model


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
    parameters, context_ids = _generate_parameters(
        function_def,
        model,
        vocab,
        context_ids
    )

    result = _build_result(prompt, function_name, parameters)
    return result


def _build_instruction_prefix(
    prompt: str, available_functions: list[FunctionDef]
) -> str:
    """Describe the available functions and frame the expected JSON answer."""
    lines = [
        "You are a function-calling assistant.",
        "Choose exactly one function that satisfies the user's request",
        "and provide its parameters.",
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
    lines.append(f'User request: "{prompt}"')
    lines.append("")
    lines.append('Answer:\n{"name": "')
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
    context_ids: list[int]
) -> tuple[dict[str, float | str], list[int]]:
    """Generate values for each parameter using appropriate grammar.

    Returns the parameters dict and the context extended with all
    generated tokens.
    """
    parameters: dict = {}
    for i, (param_name, param_type) in enumerate(
        function_def.parameters.items()
    ):
        # tell the model which parameter slot it is filling before
        # generating its value
        separator = '' if i == 0 else ', '
        label_ids = model.encode(f'{separator}"{param_name}": ')[0].tolist()
        context_ids = context_ids + label_ids

        if param_type.type == 'number':
            grammar = NumberGrammar(vocab)
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
        else:
            # remove surrounding quotes: "\"hello\"" -> "hello"
            parameters[param_name] = value_text.strip('"')
    return parameters, context_ids


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
