import argparse
import json
import sys


def parse_arguements():
    parser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description="LLM function calling tool that that translates"
                     "natural language prompts into structured function calls",
        usage="uv run python -m src [--functions_definition "
        "<function_definition_file>] [--input <input_file>] "
        "[--output <output_file>]",
        
        )

def main() -> None:

    #path = parse_arguements()

    path: str = "data/input/functions_definition.json"
    data: dict ={}
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as err:
        print(err)
        sys.exit(1)

if __name__ == "__main__":
    main()