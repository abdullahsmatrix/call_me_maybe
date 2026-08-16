from pydantic import BaseModel
from typing import Literal

class ParameterType(BaseModel):
    """Pydantic model representing "parameters" attribute of FunctionDef schema"""
    type: Literal["number", "string"]

class FunctionDef(BaseModel):
    """Pydantic model representing Function Definition schema"""
    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType

class PromptEntry(BaseModel):
    """Pydantic model representing input schema"""
    prompt: str

class FunctionCallResults(BaseModel):
    """Pydantic model representing function call result"""
    prompt: str
    name: str
    parameters: dict[str, ParameterType]


