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
