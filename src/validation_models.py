from pydantic import BaseModel, field_validator, model_validator
from typing import Any, Literal

# Some function-definition files spell types differently than the four
# canonical names we support. Map known synonyms onto our canonical types
# instead of rejecting the whole function definition over a spelling choice.
_TYPE_ALIASES = {
    "float": "number",
    "double": "number",
    "int": "integer",
    "str": "string",
    "text": "string",
    "bool": "boolean",
}


class ParameterType(BaseModel):
    """Pydantic model for parameters attribute of FunctionDef schema."""
    type: Literal["number", "string", "integer", "boolean"]

    @field_validator("type", mode="before")
    @classmethod
    def _normalize_type_alias(cls, value: Any) -> Any:
        """Map common type spellings (float/int/str/bool/...) onto the
        canonical type names before the Literal check runs."""
        if isinstance(value, str):
            key = value.strip().lower()
            return _TYPE_ALIASES.get(key, key)
        return value


class FunctionDef(BaseModel):
    """Pydantic model representing Function Definition schema."""
    name: str
    description: str
    parameters: dict[str, ParameterType]
    returns: ParameterType

    @model_validator(mode="before")
    @classmethod
    def _flatten_json_schema_parameters(cls, data: Any) -> Any:
        """Also accept the common JSON-Schema function-calling shape,
        {"parameters": {"type": "object", "properties": {...},
        "required": [...]}}, instead of only our flat
        {"parameters": {name: {"type": ...}}}.
        """
        if not isinstance(data, dict):
            return data
        params = data.get("parameters")
        if isinstance(params, dict) and "properties" in params:
            data = dict(data)
            data["parameters"] = params["properties"]
        return data


class PromptEntry(BaseModel):
    """Pydantic model representing input schema."""
    prompt: str


class FunctionCallResults(BaseModel):
    """Pydantic model representing function call result."""
    prompt: str
    name: str

    parameters: dict[str, float | str | int | bool]
