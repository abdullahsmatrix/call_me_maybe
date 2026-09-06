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
    pass


def _generate_function_name(
    available_functions: list[FunctionDef],
    model: Small_LLM_Model,
    vocab: dict,
    encoded_prompt_ids: list[int]
) -> str:
    """Use TrieMatcher grammar to generate valid function name.
    returns function name.
    """
    pass

def _generate_parameters(
    function_def: FunctionDef,
    model: Small_LLM_Model,
    vocab: dict,
    encoded_prompt_ids: list[int]
) -> dict[str, float | str]:
    """For parameters in function_def, generate a value using approproiate Grammar"""
    pass


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
