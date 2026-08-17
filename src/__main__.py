from src.arguement_parser import parse_arguements
from src.json_loader import load_json_file
import sys
from pydantic import ValidationError
from typing import Any
from src.json_parser import JsonParser
from llm_sdk import Small_LLM_Model

def main() -> None:

    args = parse_arguements()
    #Load JSON function definitions and input prompts
    functions_definitions: list[Any] = load_json_file(args.functions_definition)
    input_prompts: list[dict[str]] = load_json_file(args.input)

    #parse and validate json functions and prompts
    try:
        parsed = JsonParser(functions_definitions, input_prompts)
    except ValueError as err:
        print(err)
        sys.exit()
    print(parsed.validated_prompts)
    
    


        
    


    

if __name__ == "__main__":
    main()