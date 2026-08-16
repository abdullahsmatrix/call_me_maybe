from src.arguement_parser import parse_arguements
from src.json_loader import load_json_file
from src.models import FunctionDef, PromptEntry
import sys
from pydantic import ValidationError
from typing import Any


def main() -> None:

    args = parse_arguements()
    #Load JSON function definitions and input prompts
    functions_definitions: list[Any] = load_json_file(args.functions_definition)
    input_prompts: list[dict[str]] = load_json_file(args.input)


    validated_models: list = []
    validated_prompts: list = []
    for functions_definition in functions_definitions:
        try:
            validated_models.append(FunctionDef.model_validate(functions_definition))
        except ValidationError as err:
            print(err)
    
    for input_prompt in input_prompts:
        try:
            validated_prompts.append(PromptEntry.model_validate(input_prompt))
        except ValidationError as err:
            print(err)

    if not validated_models:
        print("No vallid function definition found")
        sys.exit()



        
    


    

if __name__ == "__main__":
    main()