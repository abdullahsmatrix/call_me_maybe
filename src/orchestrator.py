""" In this module we orchestrate the complete function call.
decoder generate tokens with grammar constraints as long as we give it the right grammar.
This module focuses on Generating a function name -> parameters a-> building JSON struct.
"""

from src.validation_models import FunctionCallResults, FunctionDef
from typing import Union
from src.grammar import TrieMatcher, NumberGrammar, StringGrammar
from src.decoder import generate_constrained
from llm_sdk import Small_LLM_Model

def call_function(
    prompt: str,
    available_functions: list[FunctionDef],
    model: Small_LLM_Model,
    vocab: dict,
    encoded_prompt_ids: list[int]
) -> FunctionCallResults:
    
    function_name = _generate_function_name(
        available_functions,
        model,
        vocab,
        encoded_prompt_ids
    )

    function_def = _find_function_def(function_name, available_functions)
    parameters = _generate_parameters(
        function_def,
        model,
        vocab,
        encoded_prompt_ids
    )

    result = _build_result(prompt, function_name, parameters)
    return result


def _generate_function_name(
    available_functions: list[FunctionDef],
    model: Small_LLM_Model,
    vocab: dict,
    encoded_prompt_ids: list[int]
) -> str:
    """Use TrieMatcher grammar to generate valid function name.
    returns function name.
    """
    function_names: list = [func.name for func in available_functions]
    grammar = TrieMatcher(function_names, vocab)
    function_name_text, _ = generate_constrained(
        model,
        encoded_prompt_ids,
        grammar,
        vocab
    )
    return function_name_text

def _generate_parameters(
    function_def: FunctionDef,
    model: Small_LLM_Model,
    vocab: dict,
    encoded_prompt_ids: list[int]
) -> dict[str, float | str]:
    """For parameters in function_def, generate a value using approproiate Grammar"""
    parameters: dict = {}
    for param_name, param_type in function_def.parameters.items():
        if param_type.type == 'number':
            grammar = NumberGrammar(vocab)
        else:
            grammar = StringGrammar(vocab)
        #generate value with constraints
        value_text, _ = generate_constrained(
            model,
            encoded_prompt_ids,
            grammar,
            vocab
        )
        #parse based on type
        if param_type.type == 'number':
            #convert string to float
            parameters[param_name] = float(value_text)
        else:
            #remove surrounding quotes: "\"hello\"" -> "hello"
            parameters[param_name] = value_text.strip('"')
    return parameters


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
        prompt= prompt,
        name = function_name,
        parameters = parameters
    )
