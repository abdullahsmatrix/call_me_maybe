from src.arguement_parser import parse_arguements
from src.json_loader import load_json_file, write_results_to_json
from src.vocab import load_or_build_vocab
from src.orchestrator import call_function
from src.json_parser import JsonParser


import sys
from pydantic import ValidationError
from typing import Any
from llm_sdk import Small_LLM_Model


def main() -> None:

    args = parse_arguements()
    #Load JSON function definitions and input prompts
    functions_definitions: list[Any] = load_json_file(args.functions_definition)
    input_prompts: list[dict[str]] = load_json_file(args.input)

    #parse and validate json functions and prompts
    try:
        parsed = JsonParser(functions_definitions, input_prompts)
        print(f"Error log: {parsed.error_log}")
    except ValueError as err:
        print(err)
        sys.exit()
    
    model = Small_LLM_Model()
    vocab: dict = load_or_build_vocab(model)
    
    


        
    


    

if __name__ == "__main__":
    main()