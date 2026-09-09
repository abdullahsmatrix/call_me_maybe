"""Grammar primitives used to constrain model token generation.

Without a grammar the model may hallucinate outputs that do not form
valid JSON. This module provides matchers (TrieMatcher) and grammars
for numbers and strings so decoding can mask invalid tokens in real
time and only allow syntactically valid continuations.
"""


class TrieMatcher:
    def __init__(self, candidates: list[str], vocab: dict) -> None:
        """Build a trie from candidate strings for prefix matching.

        Each node is a dict mapping characters to child nodes. A terminal
        node is marked by the key "is_end".
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

    def _walk(self, node: dict, text: str) -> dict | None:
        """Walk characters of text from node or return None if it falls off."""
        for char in text:
            if char not in node:
                return None
            node = node[char]
        return node

    def get_valid_token_ids(self, current_prefix: str) -> list[int]:
        """Return list of token ids that continue the given prefix.

        Tokens use BPE, so each candidate token must be walked through the
        trie rather than only considering the first character.
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
            for token_id in self.vocab['first_char_index'].get(key, []):
                token_string = self.vocab['id_to_token'][str(token_id)]
                if self._walk(current_node, token_string) is not None:
                    result.append(token_id)

        return result

    def is_complete(self, current_prefix: str) -> bool:
        """Return True if prefix matches a complete candidate string."""
        current_node = self.trie_dict
        for char in current_prefix:
            if char not in current_node:
                return False
            current_node = current_node[char]
        return bool(current_node.get("is_end", False))


class NumberGrammar:
    """Constrain partial numeric literals during generation.

    Tracks the lexical state of a number and decides which characters are
    allowed next for integers, decimals, and exponents.
    """

    def __init__(self, vocab: dict):
        """Vocab maps token characters to token ids for token resolution."""
        self.vocab = vocab
        self.STATE_CHAR_VALIDITY: dict = {
            "START": [
                "-", "0", "1", "2", "3", "4", "5",
                "6", "7", "8", "9",
            ],
            "DIGITS": [
                ".", "e", "E", "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9",
            ],
            "DECIMAL_POINT": [
                "0", "1", "2", "3", "4", "5", "6",
                "7", "8", "9",
            ],
            "FRACTION_DIGITS": [
                "e", "E", "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9",
            ],
            "EXPONENT_SIGN": [
                "+", "-", "0", "1", "2", "3", "4",
                "5", "6", "7", "8", "9",
            ],
            "EXPONENT_SIGN_DONE": [
                "0", "1", "2", "3", "4", "5", "6",
                "7", "8", "9",
            ],
            "EXPONENT_DIGITS": [
                "0", "1", "2", "3", "4", "5", "6",
                "7", "8", "9",
            ],
        }

    def _get_state(self, current_number: str) -> str:
        # Return the current grammar state for a partial numeric string.
        if not current_number:
            return "START"
        elif not any(ch in current_number for ch in ".eE"):
            return "DIGITS"
        elif current_number.endswith("."):
            return "DECIMAL_POINT"
        elif "." in current_number and not any(
            ch in current_number for ch in "eE"
        ):
            return "FRACTION_DIGITS"
        for ch in "eE":
            if ch in current_number:
                idx = current_number.index(ch)
                has_next = idx + 1 < len(current_number)
                nxt_chr = current_number[idx + 1] if has_next else None
                if nxt_chr == "+" or nxt_chr == "-":
                    return "EXPONENT_SIGN_DONE"
        if current_number.endswith("e") or current_number.endswith("E"):
            return "EXPONENT_SIGN"
        elif any(ch in current_number for ch in ("e", "E")) and (
            current_number[-1].isnumeric()
        ):
            return "EXPONENT_DIGITS"

        return "UNKNOWN"

    def _is_valid_continuation(
        self, current_number: str, token_string: str
    ) -> bool:
        """Check each char of multi-char token keeps number valid."""
        text = current_number
        for ch in token_string:
            state = self._get_state(text)
            if (
                state not in self.STATE_CHAR_VALIDITY
                or ch not in self.STATE_CHAR_VALIDITY[state]
            ):
                return False
            text += ch
        return True

    def get_valid_token_ids(self, current_number: str) -> list:
        """Return all token ids for valid chars in current state."""
        result: list = []
        state: str = self._get_state(current_number)
        if state not in self.STATE_CHAR_VALIDITY:
            return []
        valid_chars: list = self.STATE_CHAR_VALIDITY[state]
        for ch in valid_chars:
            for token_id in self.vocab['first_char_index'].get(ch, []):
                token_string = self.vocab['id_to_token'][str(token_id)]
                if self._is_valid_continuation(
                    current_number, token_string
                ):
                    result.append(token_id)
        return result

    def is_complete(self, current_number: str) -> bool:
        """Check if the number string is complete and valid.

        A number is complete when not empty, not just a sign,
        and does not end with a char requiring another digit
        or exponent component.
        """
        is_valid: bool = True
        has_sign_only = (
            not current_number
            or current_number == "+"
            or current_number == "-"
        )
        if has_sign_only:
            is_valid = False
        elif any(
            current_number.endswith(ch)
            for ch in (".", "e", "E", "e+", "E+", "e-", "E-")
        ):
            is_valid = False
        return is_valid


class StringGrammar:
    """Validate partial JSON strings during token generation."""

    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.STATE_CHAR_VALIDITY: dict = {
            "START": ['"'],
            "IN_STRING": (
                ['"', '\\']
                + [chr(i) for i in range(32, 127) if chr(i) not in '"\\']
            ),
            "ESCAPE_CHAR": ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'],
            "ESCAPE_U": list('0123456789abcdefABCDEF'),
            "ESCAPE_U_DIGITS": list('0123456789abcdefABCDEF'),
        }

        # IN_STRING allows almost the whole vocab, so precompute once
        # instead of re-validating every token character-by-character on
        # re-validating every token character-by-character on every generation
        # step.
        self._in_string_token_ids: list = self._build_in_string_token_ids()

    def _build_in_string_token_ids(self) -> list:
        """Tokens safe to emit while IN_STRING.

        No backslash, quote only as last char.
        """
        safe_ids: list = []
        for token_id_str, token_string in self.vocab['id_to_token'].items():
            if not token_string or '\\' in token_string:
                continue
            quote_idx = token_string.find('"')
            if (
                quote_idx != -1
                and quote_idx != len(token_string) - 1
            ):
                continue
            if any(not (32 <= ord(ch) <= 126) for ch in token_string):
                continue
            safe_ids.append(int(token_id_str))
        return safe_ids

    def _get_state(self, current_string: str) -> str:
        """Return the grammar state for a partial JSON string.

        Args: current_string is the string prefix generated so far.
        Returns: current state like START, IN_STRING, ESCAPE_CHAR,
            or COMPLETE.
        """
        if not current_string:
            return "START"
        if not current_string.startswith('"'):
            return "UNKNOWN"
        if current_string.endswith('"') and len(current_string) > 1:
            return "COMPLETE"

        # look for incomplete \uXXX pattern
        if '\\u' in current_string:
            last_u_idx: int = current_string.rfind('\\u')
            if last_u_idx != -1:
                hex_part: str = current_string[last_u_idx + 2:]
                if (
                    len(hex_part) < 4
                    and all(c in '0123456789abcdefABCDEF' for c in hex_part)
                ):
                    if len(hex_part) == 0:
                        return "ESCAPE_U"
                    else:
                        return "ESCAPE_U_DIGITS"

        # count trailing backslashes
        trailing_backslashes: int = 0
        for i in range(len(current_string) - 1, 0, -1):
            if current_string[i] == '\\':
                trailing_backslashes += 1
            else:
                break

        if trailing_backslashes % 2 == 1:
            # odd: last backslash is unescaped
            return "ESCAPE_CHAR"
        else:
            # even: all backslashes are escaped
            return "IN_STRING"

    def _is_valid_continuation(
        self, current_string: str, token_string: str
    ) -> bool:
        """Check each char of multi-char token keeps string valid."""
        text = current_string
        for ch in token_string:
            state = self._get_state(text)
            if (
                state not in self.STATE_CHAR_VALIDITY
                or ch not in self.STATE_CHAR_VALIDITY[state]
            ):
                return False
            text += ch
        return True

    def get_valid_token_ids(self, current_string: str) -> list:
        """Return list of valid token ids for current string state.

        Returns an empty list when the current state has no valid
        continuation.
        """
        state: str = self._get_state(current_string)
        if state not in self.STATE_CHAR_VALIDITY or state == "COMPLETE":
            return []

        if state == "IN_STRING":
            return self._in_string_token_ids

        result: list = []
        valid_chars: list = self.STATE_CHAR_VALIDITY[state]
        for ch in valid_chars:
            for token_id in self.vocab['first_char_index'].get(ch, []):
                token_string = self.vocab['id_to_token'][str(token_id)]
                if self._is_valid_continuation(
                    current_string, token_string
                ):
                    result.append(token_id)

        return result

    def is_complete(self, current_string: str) -> bool:
        """Return True if string has closing quote and is complete."""
        return self._get_state(current_string) == "COMPLETE"
