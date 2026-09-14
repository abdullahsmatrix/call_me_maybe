*This project has been created as part of the 42 curriculum by amamun.*

# call_me_maybe

## Description

This project is a function-calling CLI. You give it a plain English prompt like
"What is the sum of 2 and 3?" and a list of available functions (name,
description, parameters, types), and it spits out a structured JSON function
call: which function to use and what parameters to fill it with.

The catch, and the whole point of this project, is that the model doing the
translation is tiny: **Qwen3-0.6B**. A model that small cannot be trusted to
just "get the JSON right" on its own. Left alone it will happily hallucinate
broken brackets, unterminated strings, garbage numbers, or function names that
don't even exist. So instead of hoping for the best, this project never lets
the model choose freely in the first place. At every single token it's about
to generate, we ask "out of the whole vocabulary, which tokens are even
legal right now?", mask everything else to `-inf`, and only then let the model
pick. The model still decides *what* to say (which function, what values),
but it is structurally incapable of saying it wrong syntax-wise.

This technique is usually called **constrained decoding** (or grammar-constrained
generation). Full technical breakdown with diagrams lives in
[TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) if you want
the deep dive — this README covers the essentials. For how the parameter
values in particular were made accurate (and the bugs found along the
way), see [PARAMETER_ACCURACY.md](PARAMETER_ACCURACY.md).

## Instructions

This project uses [uv](https://docs.astral.sh/uv/) for dependency management,
and everything is wired through the Makefile so you don't need to remember any
commands.

**Install dependencies:**
```bash
make install
```

**Run the program:**
```bash
make run
```
This runs `uv run python3 -m src` under the hood. By default it reads function
definitions from `data/input/functions_definition.json`, prompts from
`data/input/function_calling_tests.json`, and writes results to
`data/output/function_calling_results.json`.

You can point it at your own files too:
```bash
uv run python3 -m src --functions_definition path/to/functions.json --input path/to/prompts.json --output path/to/results.json
```

**First run note:** the very first time you run this it will download
Qwen3-0.6B from the Hugging Face Hub (a few hundred MB) and build a
vocabulary cache under `data/cache/`. Every run after that is fast because it
reuses the cache. If you ever change the model or something looks off, just
delete `data/cache/cache.json` and it'll rebuild itself.

**Other Makefile targets:**
```bash
make lint          # flake8 + mypy
make lint-strict    # same but mypy --strict
make clean          # remove __pycache__ / mypy / pytest caches
make cache_clean    # clear the uv package cache
```

## Algorithm Explanation (Constrained Decoding)

The core idea is a loop that repeats for every single token the model
generates:

1. Ask the model for raw next-token logits.
2. Ask a **grammar** object which token ids are legal right now, given
   everything generated so far.
3. Set every other token's logit to `-inf`.
4. Take the argmax — the model can only "choose" among what's left.
5. Append the picked token and repeat until the grammar says we're done.

Because invalid tokens are removed *before* the model picks, there is no
path for the model to ever emit broken syntax. It's not that the model is
well-behaved it's just that it's not given the option to misbehave.

Four grammars exist, one for each kind of value we ever need to generate:

- **`TrieMatcher`** — used for function names (and for `true`/`false`).
  Builds a prefix tree out of the
  fixed list of available function names, so at any point only characters
  that continue an actual candidate name are legal. Since the tokenizer
  works in multi-character BPE chunks and not single letters, we walk every
  character of a candidate token through the trie, not just its first
  letter, otherwise a token like `"nd"` could sneak in right after `"f"` and
  produce garbage like `"fnd"`.
- **`NumberGrammar`** — a small state machine mirroring the JSON number
  grammar (sign, digits, decimal point, fraction digits, exponent sign,
  exponent digits). At each state only a specific set of characters are
  legal, so you literally cannot generate `"1..2"` or `"1e"` as a final
  answer.
- **`IntegerGrammar`** — the same idea, narrowed: an optional sign followed
  by digits, with no decimal point or exponent ever allowed.
- **`StringGrammar`** — same idea but for JSON strings: tracks opening quote,
  string body, escape sequences (`\n`, `\t`, `\"`, `\uXXXX`), and closing
  quote. A string is complete only once an unescaped closing quote has been
  emitted. Today this is a **fallback** — string parameters are normally
  filled by candidate selection instead (see below).

**String parameters work differently.** Rather than generating characters
freely, we extract every word, quoted phrase, and after-the-last-colon
remainder from the prompt, and let the model choose among *those literal
spans*. Every correct string value in this task is a substring of the
request, so the model is only ever asked "which span?", never "which
characters?". That makes hallucination and repetition loops impossible by
construction. The full reasoning is in
[PARAMETER_ACCURACY.md](PARAMETER_ACCURACY.md).

All the JSON scaffolding around these generated values — the braces, colons,
commas, key names — is never generated by the model at all. It's written
directly by the orchestrator, deterministically. The model only ever fills
in the blanks, and only with grammar-approved characters.

## Design Decisions

**Two-phase generation per prompt.** For every prompt, we first generate the
function name (via `TrieMatcher`), then fill each parameter in order:
`NumberGrammar` / `IntegerGrammar` for numerics, `TrieMatcher` for booleans,
and candidate selection for strings. This mirrors how you'd naturally fill
out a form: pick which form first, then fill in the fields.

**Threaded context between generation steps.** Early on, each generation
step (function name, then each parameter) only ever saw the raw user prompt
in isolation — the model had no idea what function it had already picked or
what earlier parameters it had already committed to. We fixed this by
threading a growing sequence of token ids through the whole call: the
context for parameter 2 includes everything generated for parameter 1, the
function name, and so on. Each parameter is also preceded by an explicit
`"param_name": ` label injected into the context so the model knows exactly
which slot it's filling.

**Instruction prefix listing the available functions.** Originally the model
was just shown the bare prompt with zero information about what functions
existed. Unsurprisingly, it picked almost randomly. Hence we built an explicit
instruction block before generation starts, listing every function's name,
parameter names/types, and description, followed by the user's actual
request and the opening of the answer (`{"function": "`). This alone was
the single biggest improvement to function-selection accuracy.

The scaffold key is `"function"`, not `"name"`, on purpose: a prompt
containing a template placeholder like `{name}` collides with a `{"name": "`
scaffold — the model reads it as that placeholder being filled in and starts
predicting a person's name instead of a function name. The keys in the
emitted JSON are unaffected, since those come from the pydantic result model.

**Deterministic JSON structure, generated values only.** We never ask the
model to produce a brace, a comma, or a key name. Those are 100% certain to
be correct if we just write them ourselves, so there's no reason to burn
model calls (or risk mistakes) on them. The model is only ever asked to fill
in things that actually vary: the function name and the parameter values.

**Vocabulary caching + precomputed safe-token sets.** Decoding the tokenizer
vocabulary from BPE encoding and building a `first_char_index` lookup is a
one-time cost, so it's cached to disk after the first run. Similarly,
`StringGrammar`'s "inside a string" state legally allows almost the entire
vocabulary checking every token character-by-character on every single
generation step was way too slow, so the set of "safe" tokens for that state
is computed once up front and reused.

## Performance Analysis

**Reliability (syntax):** 100%. Every single output produced by this
pipeline is valid JSON matching the expected schema. A valid function name
from the provided list, and parameter values that are properly formed JSON
numbers/strings. This is mechanically guaranteed by the grammars.

**Accuracy (semantics):** measured against the ground truth in
`data/input/function_calling_corrections.json`, the pipeline currently gets
**11/11 function names and 11/11 parameter sets**. Getting there took
several distinct fixes — constraining string values to literal spans of the
prompt, scoring whole candidates instead of greedily picking a first token,
scoring them *terminated*, and assigning candidates to parameter slots
globally rather than left to right. Each of those is explained, with the
measurements that motivated it, in
[PARAMETER_ACCURACY.md](PARAMETER_ACCURACY.md).

Sanity-checked against overfitting: on an unseen function set using
JSON-Schema-shaped parameters and `str`/`float`/`int` type aliases, with
prompts never seen before, all 3/3 came out correct.

**Speed:** the biggest cost by far is that the model has no KV-cache, so
every single generated token requires a full forward pass over the entire
sequence generated so far. This means later tokens in a long generation are
more expensive than earlier ones. A full run over the 11 test prompts takes
roughly **3 minutes** on a Mac after the vocabulary cache is warm — inside
the subject's 5-minute budget, but slower than pure greedy decoding, because
scoring string candidates costs a forward pass per candidate token rather
than one walk. That is the price paid for the accuracy above. First run is
slower still, because it also downloads the model and builds the vocab cache.

## Challenges Faced

**BPE tokens aren't single characters.** My first version of the grammars
only checked a candidate token's *first* character against what was legal,
then let the whole token through. Since the tokenizer merges multiple
characters into single tokens, this let invalid multi-character tokens
sneak in mid-generation (e.g. generating `"fnd"` when no function name even
has that substring). Fixed by walking every character of a candidate token
through the grammar/trie before accepting it, not just the first one.

**That fix made string generation painfully slow.** Once every candidate
token was checked character-by-character, `StringGrammar`'s "inside a
string" state (which legally allows something like 95 different starting
characters) ended up re-scanning almost the entire ~150k-token vocabulary on
every single generated character. It technically worked, but it took so
long it looked like the program had hung. Fixed by precomputing, once per
grammar instance, the full set of tokens that are always safe to emit while
inside a string, instead of re-deriving that from scratch at every step.

**Vocabulary key types silently changed between runs.** The very first time
the vocab is built, its `id_to_token` dict has integer keys in memory. Once
it gets written to the cache file and read back with `json.load` on a later
run, JSON forces all dict keys to become strings — so the exact same lookup
that worked on a fresh build would throw a `KeyError` on every cached run.
Fixed by normalizing token ids to strings consistently on both build and
cache-load paths.

**PyTorch tensor shape mismatches.** `model.encode()` returns a 2-D tensor
(`[[id1, id2, ...]]`), and naively calling `.tolist()` on it kept the nested
list structure, which then broke `model.get_logits_from_input_ids()`
downstream (it wraps whatever it's given in another `torch.tensor([...])`,
so a nested list produced a jagged/invalid tensor). Fixed by explicitly
flattening with `model.encode(text)[0].tolist()`.

**The model always answering with the same numbers.** "What is the sum of
265 and 345" kept coming back as `a: 2.0, b: 3.0` — the model reproducing a
canonical "2 + 3" example instead of reading the actual numbers. The
instruction prefix had at one point hardcoded few-shot examples using
specific function names; those were removed, since the subject explicitly
forbids hardcoding around the provided examples. What actually fixed value
extraction was the redesign in
[PARAMETER_ACCURACY.md](PARAMETER_ACCURACY.md).

**Multi-digit numbers were cut off after one digit.** `12345` came out as
`1`. `NumberGrammar.is_complete("1")` is correct — `1` *is* a complete JSON
number — but the decode loop treated "this could end here" as "end here",
and a number has no terminator character to say otherwise. Fixed by checking
the model's *unconstrained* top choice at that point: if it still wants a
digit, keep going; if it's moved on to a comma or brace, stop.

**The program appeared to freeze.** One prompt took 318 seconds. Greedy
decoding had fallen into a self-reinforcing repetition loop inside a string
value and never emitted a closing quote, burning the whole iteration budget
— with no KV-cache, cost grows roughly quadratically. Fixed with a
repetition guard that detects a short token cycle repeating and bails out
(318s → 40s). The underlying cause went away entirely once string values
became span selection rather than free generation.

## Testing Strategy

Given the "must not rely on the model spontaneously producing correct JSON"
constraint, testing focused on validating the **grammar layer** in
isolation, since that's the part that has to be provably correct regardless
of what the model does:

- Manually walking through `NumberGrammar` and `StringGrammar`'s state
  transitions against known-valid and known-invalid strings (`"2.5"`,
  `"2."`, `"-"`, `"1e"`, `"1e+10"`, unterminated strings, escaped quotes,
  `\uXXXX` sequences) to confirm `is_complete()` and `get_valid_token_ids()`
  agree with the JSON grammar spec.
- Running the full CLI end-to-end against the provided
  `function_calling_tests.json` and inspecting the output JSON for
  structural validity (every result parses as JSON, every `name` is one of
  the declared functions, every parameter value matches its declared type).
- Deliberately feeding malformed function definitions and prompts through
  `JsonParser` to confirm invalid entries are skipped individually (with a
  logged validation error) instead of crashing the whole run.
- Iteratively debugging by printing the accumulated text at each decoding
  step, which is how the "only generates the first digit of multi-digit
  numbers" and "re-scans the whole vocab per character" issues were actually
  caught — both looked fine on the surface until traced token-by-token.
- Scoring the full pipeline against the ground truth in
  `data/input/function_calling_corrections.json`, comparing the generated
  `name` and `parameters` field-by-field. This turned "the output looks
  roughly right" into a number, which is what made it possible to tell real
  fixes from plausible-sounding ones — several changes that seemed sensible
  turned out to score the same or worse and were reverted.
- Printing the model's raw, *unmasked* top-k logits at a failing generation
  step. Seeing what the model wanted before any constraint was applied is
  what identified the `{name}` placeholder / `{"name": "` scaffold collision
  (see [PARAMETER_ACCURACY.md](PARAMETER_ACCURACY.md) §4).
- Re-running against an unseen function set with different parameter shapes
  and type spellings, to check the solution generalises rather than fitting
  the provided examples.

## Example Usage

Given a `functions_definition.json` like:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

And a prompt file like:
```json
[
  { "prompt": "What is the sum of 2 and 3?" }
]
```

Running:
```bash
make run
```

Produces a `function_calling_results.json` like:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  }
]
```

## Resources

- [JSON grammar specification (json.org)](https://www.json.org/json-en.html) —
  the actual grammar our `NumberGrammar`/`StringGrammar` state machines are
  modeled after.
- [Qwen3 model card (Hugging Face)](https://huggingface.co/Qwen/Qwen3-0.6B) —
  the model used for generation in this project.
- [Constrained decoding / grammar-based sampling](https://huggingface.co/docs/transformers/main/en/internal/generation_utils) —
  background on masking logits to force structured output, the general
  family of technique this project implements by hand.
- [Byte-Pair Encoding (BPE) tokenization](https://huggingface.co/learn/nlp-course/chapter6/5) —
  needed to understand why grammars have to validate whole tokens, not just
  single characters.
- [Pydantic documentation](https://docs.pydantic.dev/) — used for all input/output
  schema validation in this project.
- [Parsing Incrementally for Constrained Auto-Regressive Decoding
  from Language Models](https://aclanthology.org/2021.emnlp-main.779.pdf) - Torsten Scholak and Nathan Schucher and Dzmitry Bahdanau
- [Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation - Luca Beurer-Kellner, Marc Fischer, Martin      Vechev](https://files.sri.inf.ethz.ch/website/papers/beurerkellner2024domino.pdf)
- [Tokenizers](https://huggingface.co/docs/tokenizers/index)
- [ArgParse Documentation](https://docs.python.org/3/library/argparse.html)


**How AI was used:** This project was built with GitHub Copilot (Claude Sonnet)
acting as a mentor/pair-programmer rather than an autocomplete tool. For
most of the implementation (the grammar classes, the decoding loop, the
orchestrator), I wrote the code myself after Copilot explained the design
and pointed out what a step needed to accomplish, then reviewed what I wrote
line by line, catching real bugs like the first-character-only trie
matching, the `.append()` vs `.extend()` mistake, and the trie resetting to
root on every loop iteration. Copilot was also used to directly implement
a handful of things: the debugging/performance-fix pass on the grammars and
decoder near the end of the project (whole-token validation, precomputing
safe string tokens, threading context between generation steps). This README.md
plus the Technical Documentation was made in collaboration with Claude Haiku.
No project logic was generated wholesale without me understanding what it does
such as the constrained decoding architecture, the choice of grammars, and the
overall design were worked out through back-and-forth discussion, not generated blind.