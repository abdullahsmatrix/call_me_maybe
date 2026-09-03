class TrieMatcher():
    def __init__(self, candidates: list[str], vocab: dict) -> None:
        self.candidates = candidates
        self.vocab = vocab
        self.trie_dict: dict = {}

        for candidate in candidates:
            current_node = trie_dict
            for char in candidate:
                if char not in current_node:
                    current_node[char] = {}
                current_node = current_node[char]
            current_node["is_end": True]

        print(self.trie_dict)