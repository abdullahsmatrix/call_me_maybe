#from grammar.py import TrieMatcher
# # from models import FunctionDef
# from arguement_parser import parse_arguements
# from json_loader import load_json_file
# from models import FunctionDef
# args = parse_arguements()

# function_defs = load_json_file(args.functions_definition)

# # functions = FunctionDef.model_validate_json(function_defs)

# # print(functions)
# # Parse the first function definition from the JSON array
# functions = [FunctionDef(**func) for func in function_defs]

# for func in functions:
#     print(func.parameters)
#     print("\n")


# vocab = {"aaaaĠee": 123, "bbbbĊff": 456}

# inverse_vocab = {}

# for k, v in vocab.items():
#     # 1. Clean the key string and assign it to a new variable
#     cleaned_key = k.replace('Ġ', ' ').replace('Ċ', '\n')
    
#     # 2. Map the value to the newly cleaned key in inverse_vocab
#     inverse_vocab[clean] = cleaned_key

# print(inverse_vocab)
# # Output: {123: 'aaaa ee', 456: 'bbbb\nff'}



# print(voca)


# vocab = {"aaaaĠaaa": 123, "bbbbĊbb": 321}

# inverse_vocab = {}
# for k, v in vocab.items():
#     #k = "hhĠjjj"
#     if 'Ġ' in k:
#         inverse_vocab[v] = k.replace('Ġ', ' ')
#     elif 'Ċ' in k:
#         inverse_vocab[v] = k.replace('Ċ', '\n')
# print(inverse_vocab)

# candidates = ["fn_add_numbers", "fn_greet"]


# trie_dict: dict = {}

# for candidate in candidates:
#     current_node = trie_dict
#     for char in candidate:
#         if char not in current_node:
#             current_node[char] = {}
#         current_node = current_node[char]
#     current_node["is_end"] = True

# print(trie_dict)


# current_number: str = "1"

# if _ not in current_number for _ in [".", "e", "E"]:
#     print("DIGITS")
