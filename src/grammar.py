"""
This Module is grammar primitatives. Without grammar the model hallucinates.
The output is not certain that it will give JSON schema. We apply 
TrieMatcher for function names at each token generation step. eg; after
generating "f" only tokens that start with "n_" are allowed to continue.
The logits for invalid tokens are masked to negative INF. Model cant deviate.
It is forced to generate guaranteed ouput.
"""

class TrieMatcher():
    def __init__(self, candidates: list[str], vocab: dict) -> None:
        """TrieMatcher Initializer. We create a tree data structure using
        dictionary. Each node have children node for next possible chars.
        if it is at the end, final node is marked as "is_end" = True.

        eg; {"f": {"n": {"_": {"a": {...},
                               "g": {...},
                        }     }
                   }
            }
            
        """

        self.candidates = candidates
        self.vocab = vocab
        self.trie_dict: dict = {}

        for candidate in candidates:
            current_node = self.trie_dict
            for char in candidate:
                if char not in current_node:
                    current_node[char] = {}
                current_node = current_node[char]
            current_node["is_end"] = True
    
    def get_valid_token_ids(self, current_prefix: str) -> list[int]:
        """In this function we get a list of valid token Ids. Lets say we are
        in "fn_". our functions are "fn_add_numbers" and "fn_greet". The func
        returns tokend IDs for "a" and "g" as a list.
        """
        result: list = []
        current_node = self.trie_dict
        for char in current_prefix:
            if char not in current_node:
                return []
            current_node = current_node[char]
        for key in current_node.keys():
            if key == "is_end":
                continue
            result.extend(self.vocab['first_char_index'][key])

        return result
    
    def is_complete(self, current_prefix: str) -> bool:
        """checker function to see if the prefix is complete candidate"""
        current_node = self.trie_dict
        for char in current_prefix:
            if char not in current_node:
                return False
            current_node = current_node[char]
        return current_node.get("is_end", False)
            

class NumberGrammar():
    def __init__(self, vocab: dict):
        self.vocab = vocab
    
    def _get_state(self, current_number: str) -> str:
        if not current_number:
            return "START"
        elif not any(ch in current_number for ch in ".eE"):
            return "DIGITS"
        elif current_number.endswith("."):
            return "DECIMAL_POINT"
        elif "." in current_number and not any(ch in current_number for ch in "eE"):
            return "FRACTION_DIGITS"
        for ch in "eE":
            if ch in current_number:
                idx = current_number.index(ch)
                nxt_chr = current_number[idx + 1] if idx+1 < len(current_number) else None
                if nxt_chr == "+" or nxt_chr == "-":
                    return "EXPONENT_SIGN_DONE"
        if current_number.endswith("e") or current_number.endswith("E"):
            return "EXPONENT_SIGN"
        elif any(ch in current_number for ch in ("e", "E")) and current_number[-1].isnumeric():
            return "EXPONENT_DIGITS"
        
        return "UNKNOWN"
            
    def get_valid_token_ids(self, current_number: str) -> list:

        result: list = []
        state_char_validity: dict = {
            "START": ["-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "DIGITS": [".", "e", "E", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "DECIMAL_POINT": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "FRACTION_DIGITS": ["e", "E", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "EXPONENT_SIGN": ["+", "-", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "EXPONENT_SIGN_DONE": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            "EXPONENT_DIGITS": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        }

        state: str = self._get_state(current_number)
        if state not in state_char_validity:
            return []
        valid_chars: list = state_char_validity[state]
        for ch in valid_chars:
            token_ids = self.vocab['first_char_index'].get(ch, [])
            result.extend(token_ids)
        return result

    def is_complete(self, current_number: str) -> bool:
        is_valid: bool = True
        if not current_number or current_number == "+" or current_number == "-":
            is_valid = False
        elif any(current_number.endswith(ch) for ch in (".", "e", "E", "e+", "E+", "e-", "E-")):
            is_valid = False
        return is_valid
