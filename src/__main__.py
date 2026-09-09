from src.arguement_parser import parse_arguements
from src.json_loader import load_json_file, write_results_to_json
from src.vocab import load_or_build_vocab
from src.orchestrator import call_function
from src.json_parser import JsonParser


import sys
from typing import Any
from llm_sdk import Small_LLM_Model


def main() -> None:
    results: list = []

    args = parse_arguements()
    # Load JSON function definitions and input prompts
    functions_definitions: list[Any] = load_json_file(
        args.functions_definition
    )
    input_prompts: list[dict[str]] = load_json_file(args.input)

    # Parse and validate json functions and prompts
    try:
        parsed = JsonParser(functions_definitions, input_prompts)
        if parsed.error_log:
            print(f"Error log: {parsed.error_log}")
    except (ValueError, Exception) as err:
        print(err)
        sys.exit(1)

    model = Small_LLM_Model()
    vocab: dict = load_or_build_vocab(model)

    for prompt_entry in parsed.validated_prompts:
        prompt_text = prompt_entry.prompt

        result = call_function(
            prompt=prompt_text,
            available_functions=parsed.validated_functions,
            model=model,
            vocab=vocab
        )

        results.append(result)
    write_results_to_json(results, args.output)

    print(f"Processing complete! {len(results)} function calls generated.")


if __name__ == "__main__":
    main()
