# from models import FunctionDef
from arguement_parser import parse_arguements
from json_loader import load_json_file
from models import FunctionDef
args = parse_arguements()

function_defs = load_json_file(args.functions_definition)

# functions = FunctionDef.model_validate_json(function_defs)

# print(functions)
# Parse the first function definition from the JSON array
functions = [FunctionDef(**func) for func in function_defs]

for func in functions:
    print(func.parameters)
    print("\n")