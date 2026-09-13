from src.validation_models import PromptEntry, FunctionDef
from pydantic import ValidationError
from typing import Any


class JsonParser:
    def __init__(
        self,
        functions_definitions: list[Any],
        input_prompts: list[dict[str, Any]],
    ) -> None:
        self.functions_definitions = functions_definitions
        self.input_prompts = input_prompts
        self.validated_functions: list = []
        self.validated_prompts: list = []
        self.error_log: list = []
        definitions = enumerate(self.functions_definitions)
        for index, functions_definition in definitions:
            # checks if the JSON object is valid against pydantic base
            # model and appends to the list
            try:
                self.validated_functions.append(
                    FunctionDef.model_validate(functions_definition)
                )
            except ValidationError as err:
                name = (
                    functions_definition.get("name", "<unnamed>")
                    if isinstance(functions_definition, dict)
                    else "<unnamed>"
                )
                self.error_log.append(
                    f"Skipped function definition #{index} "
                    f"('{name}'): {err}"
                )

        for index, input_prompt in enumerate(self.input_prompts):
            try:
                self.validated_prompts.append(
                    PromptEntry.model_validate(input_prompt)
                )
            except ValidationError as err:
                self.error_log.append(
                    f"Skipped input prompt #{index}: {err}"
                )

        if not self.validated_functions:
            msg = "No valid function definition found"
            raise ValueError(msg)

        if not self.validated_prompts:
            raise ValueError("No valid input prompt given")
