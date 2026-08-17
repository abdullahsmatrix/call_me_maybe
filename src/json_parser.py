from src.validation_models import ParameterType, PromptEntry, FunctionDef, FunctionCallResults

class JsonParser():
    def __init__(self, functions_definitions: list[Any], input_prompts: list[dict[str]]) -> None:
        self.functions_definitions = functions_definitions
        self.input_prompts = input_prompts
        self.validated_functions: list = []
        self.validated_prompts: list = []

        for functions_definition in self.functions_definitions:
            """
            checks if the JSON object is valid against pydantic base model and appends to the list
            """
            self.validated_functions.append(FunctionDef.model_validate(functions_definition))
        
    
        for input_prompt in self.input_prompts:
            self.validated_prompts.append(PromptEntry.model_validate(input_prompt))


        if not self.validated_functions:
            raise ValidationError("No vallid function definition found")

        if not self.validated_prompts:
            raise ValueError("No valid input prompt given")