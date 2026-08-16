import json
import sys

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
