from pathlib import Path
import json


BPE_DECODE_TABLE = {
    "Ġ": " ",
    "Ċ": "\n",
}

def load_or_build_vocab(model) -> dict:

    cache_path = Path("data/cache/cache.json")

    
    try:
        with cache_path.open("r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as err:
        print(f"Error: {err}. Building vocab...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        vocab: dict = build_vocab(model)
        with cache_path.open("w") as file:
            json.dump(vocab, file)
            return vocab
    
def build_vocab(model) -> dict:
    vocab_path = model.get_path_to_vocab_file()
    with open(vocab_path, "r") as file:
        vocab = json.load(file)

    inverse_vocab = {}
    for k, v in vocab.items():
        if 'Ġ' in k:
            k = k.replace('Ġ', ' ')
            inverse_vocab[v] = k
        elif 'Ċ' in k:
            k = k.replace('Ċ', '\n')
            inverse_vocab[v] = k




