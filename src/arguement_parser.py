import argparse

def parse_arguements() -> argparse.Namespace:
    """Commandline arguement parser. Parses arguements and raises errors gracefully"""
    
    parser = argparse.ArgumentParser(
        prog="call_me_maybe",
        description="LLM function calling tool that that translates "
                     "natural language prompts into structured function calls",
        )

    parser.add_argument(
        "--functions_definition",
        help = "Declare path to JSON file containing function definitions",
        default= "data/input/functions_definition.json",
        )
    parser.add_argument(
        "--input",
        help="Declare path to input file containing the prompts",
        default="data/input/function_calling_tests.json",
        )
    parser.add_argument(
        "--output",
        help="Declare path to JSON output file",
        default="data/output/function_calling_results.json"
        )

    args = parser.parse_args()
    return args