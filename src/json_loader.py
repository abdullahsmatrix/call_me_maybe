import json
import sys
from .validation_models import FunctionCallResults
from pathlib import Path

def load_json_file(filename: str) -> dict:
    """Loads JSON file and returns data as a python dictionary.
        Handles error gracefully and prints on terminal.
    """

    try:
        with open(filename, "r") as file:
            return json.load(file)

    except FileNotFoundError as err:
        print(err)
        sys.exit()
    except PermissionError as err:
        print(err)
        sys.exit()
    except json.JSONDecodeError as err:
        print(err)
        sys.exit()
    except Exception as err:
        print(f"An error occured. Details: {err}")
        sys.exit()


def write_results_to_json(results: list[FunctionCallResults], output_path: str) -> None:
    
    result_dict: list[dict] = []
    
    path_object = Path(output_path)
    
    path_object.parent.mkdir(parents=True, exist_ok=True)
    
    for result in results:
        result_dict.append(result.model_dump())
    
    try:
        with path_object.open("w") as file:
            json.dump(result_dict, file, indent=2)
    except (PermissionError, OSError, IOError) as err:
        print(err)
        sys.exit()