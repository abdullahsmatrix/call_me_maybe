import json
import sys
from typing import Any
from src.validation_models import FunctionCallResults
from pathlib import Path


def load_json_file(filename: str) -> Any:
    """Load a JSON file and return its contents as a dict.

    Errors are printed to stderr and the process exits with status 1.
    """

    try:
        with open(filename, "r") as file:
            return json.load(file)

    except FileNotFoundError as err:
        print(err)
        sys.exit(1)
    except PermissionError as err:
        print(err)
        sys.exit(1)
    except json.JSONDecodeError as err:
        print(f"Decode error in {Path(filename).name}: {err}")
        sys.exit(1)
    except Exception as err:
        print(f"An error occurred. Details: {err}")
        sys.exit(1)


def write_results_to_json(
    results: list[FunctionCallResults], output_path: str
) -> None:
    result_dict: list[dict] = []

    path_object = Path(output_path)
    path_object.parent.mkdir(parents=True, exist_ok=True)

    for result in results:
        result_dict.append(result.model_dump())

    try:
        with path_object.open("w") as file:
            json.dump(result_dict, file, indent=2)
    except PermissionError as err:
        print(f"Cannot write to file, {err}")
        sys.exit(1)
    except (OSError, IOError) as err:
        print(err)
        sys.exit(1)
