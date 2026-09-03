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
            
