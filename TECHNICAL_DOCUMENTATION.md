# call_me_maybe — Technical Documentation

## 1. Overview

`call_me_maybe` is a command-line tool that translates natural-language prompts
into structured function calls (name + parameters), expressed as JSON.

The distinguishing constraint of this project is that the language model doing
the translation is intentionally tiny — **Qwen3-0.6B**. A model this small
cannot be trusted to spontaneously emit syntactically correct JSON on its own;
left unconstrained it will hallucinate stray characters, malformed numbers,
unterminated strings, or invalid function names.

Instead of hoping the model gets the syntax right, the system **guarantees**
it through **constrained decoding**: at every single token-generation step,
the set of tokens the model is allowed to produce is restricted to only those
that keep the output on a path toward valid JSON. Tokens that would break
syntax are masked out (their probability is forced to negative infinity)
*before* the model ever gets to choose, so the model is structurally
incapable of producing invalid output — regardless of how small or
under-trained it is.

The model is still responsible for the *semantic* decisions (which function
to call, what values to fill in), but never for punctuation, quoting, or
structure. Those are always either forced by a grammar or written
deterministically by the program itself.

## 2. Core Design Principle

> Never let the model choose freely. Always narrow its choices down to the
> set of tokens that are provably valid at this exact point in the output.

This is implemented as a repeating cycle:

1. Ask the model for its raw next-token probabilities (logits).
2. Ask a **grammar** object which tokens are allowed right now, given
   everything generated so far.
3. Mask every other token's probability to `-∞`.
4. Let the model pick the highest-probability token from what remains.
5. Append it to the output and repeat until the grammar says the value is
   complete.

Because step 3 removes every invalid possibility before step 4's selection
happens, the output cannot deviate from the grammar no matter what the model
"wants" to generate.

## 3. High-Level Architecture

```mermaid
flowchart TD
    subgraph Input
        A1[functions_definition.json]
        A2[function_calling_tests.json]
    end

    A1 --> B[Validation Layer<br/>validation_models.py + json_parser.py]
    A2 --> B

    B --> C[Vocabulary Layer<br/>vocab.py]
    C --> D[Grammar Layer<br/>grammar.py]
    D --> E[Decoding Engine<br/>decoder.py]
    E --> F[Orchestrator<br/>orchestrator.py]
    F --> G[CLI Entry Point<br/>__main__.py]
    G --> H[function_calling_results.json]

    subgraph Model
        M[Qwen3-0.6B<br/>Small_LLM_Model]
    end

    E <--> M
    C <--> M
```

## 4. Module Responsibilities

| Module | Responsibility |
|---|---|
| `arguement_parser.py` | Parses CLI flags: paths to function definitions, input prompts, and output file. |
| `validation_models.py` | Pydantic schemas describing the shape of function definitions, prompts, and results. Acts as the single source of truth for "what does valid input/output look like." |
| `json_loader.py` | Reads JSON files from disk and writes the final results back out, with defensive error handling (missing file, bad permissions, malformed JSON). |
| `json_parser.py` | Validates every loaded function definition and prompt against the Pydantic schemas, discarding invalid entries individually (with a warning) rather than failing the whole run. |
| `vocab.py` | Loads the model's tokenizer vocabulary once, decodes it from BPE byte-level encoding into normal readable text, and builds a lookup index (`first_char_index`) that maps a starting character to every token that begins with it. Cached to disk so this expensive step only happens once. |
| `grammar.py` | Defines the constraint rules ("grammars") that decide which tokens are legal at any given point: matching a fixed function name, building a valid number, or building a valid quoted string. |
| `decoder.py` | The generation loop itself: repeatedly asks the model for logits, asks the active grammar what's allowed, masks everything else, and picks a token — until the grammar reports the value is complete. |
| `orchestrator.py` | Coordinates a full function call: builds the instruction context (available functions + user prompt), drives function-name generation, then drives parameter-value generation one field at a time, threading the growing context between steps. |
| `__main__.py` | Wires everything together: parses args, loads and validates files, initializes the model and vocabulary, runs the orchestrator over every prompt, and writes the results. |

## 5. The Grammar Layer in Detail

Three grammars exist, one per kind of value the model is ever asked to produce.
Each one exposes the same two-method contract, so the decoding engine can work
with any of them interchangeably:

- `get_valid_token_ids(current_text)` — given what has been generated so far,
  return every token id that is legal to generate next.
- `is_complete(current_text)` — has the value reached a valid, finished state?

### 5.1 TrieMatcher — function names

Builds a prefix tree (trie) out of the fixed list of available function
names. At any point during generation, the current prefix locates a node in
the trie; only branches that exist from that node are legal continuations.
This guarantees the model can only ever produce one of the known function
names — never a fabricated one.

Because the tokenizer uses multi-character sub-word tokens (not single
characters), a candidate token is only accepted if walking through *every*
character of that token keeps a valid path through the trie — not just its
first character. This prevents multi-character tokens from jumping the model
onto an invalid branch mid-name.

### 5.2 NumberGrammar — numeric parameters

Modeled as a small state machine that mirrors the JSON number grammar:

```mermaid
stateDiagram-v2
    [*] --> START
    START --> DIGITS: digit or "-"
    DIGITS --> DIGITS: digit
    DIGITS --> DECIMAL_POINT: "."
    DIGITS --> EXPONENT_SIGN: e/E
    DECIMAL_POINT --> FRACTION_DIGITS: digit
    FRACTION_DIGITS --> FRACTION_DIGITS: digit
    FRACTION_DIGITS --> EXPONENT_SIGN: e/E
    EXPONENT_SIGN --> EXPONENT_SIGN_DONE: "+"/"-"
    EXPONENT_SIGN --> EXPONENT_DIGITS: digit
    EXPONENT_SIGN_DONE --> EXPONENT_DIGITS: digit
    EXPONENT_DIGITS --> EXPONENT_DIGITS: digit
    DIGITS --> [*]: complete
    FRACTION_DIGITS --> [*]: complete
    EXPONENT_DIGITS --> [*]: complete
```

At each state, only a specific handful of characters are legal (e.g. only
digits right after a decimal point). A value is considered complete once it
is non-empty and doesn't end on a state that's still expecting more
characters (a trailing sign, a trailing decimal point, a trailing exponent
marker).

### 5.3 StringGrammar — string parameters

Also a state machine, mirroring JSON's quoted-string grammar: it tracks
whether it's before the opening quote, inside the string body, in the middle
of an escape sequence (`\n`, `\t`, `\"`, …), or inside a `\uXXXX` unicode
escape collecting exactly four hex digits. The string is complete once a
closing, unescaped quote has been produced.

Because the "inside the string" state legally allows the overwhelming
majority of printable characters, checking every vocabulary token against it
character-by-character on every generation step would be prohibitively slow.
Instead, the set of tokens that are always safe to emit while inside a string
(no backslash, no interior quote) is computed **once** up front, and reused
for the entire run.

## 6. The Decoding Engine

`generate_constrained` is the single implementation of the generation loop
described in section 2. It is grammar-agnostic — it does not know or care
whether it's generating a function name, a number, or a string; it simply
asks whichever grammar it was given what's currently legal.

Two safety mechanisms bound its runtime:

- If the grammar ever reports that *nothing* is currently legal (a dead end),
  generation stops immediately rather than continuing to burn model calls.
- A maximum iteration count caps how many tokens can be generated for any
  single value, in case the model never happens to pick a "completing" token.

Because the underlying model has no key-value cache, every single generated
token requires a full forward pass over the entire sequence generated so far.
This is the dominant cost driver of the whole pipeline and the reason the
grammars are written to be as cheap as possible per step.

## 7. The Orchestrator: Building One Function Call

A single prompt is turned into a function call in three stages, all sharing
one growing sequence of token ids (`context_ids`) so that later stages can
"see" everything decided in earlier stages:

```mermaid
sequenceDiagram
    participant U as User Prompt
    participant O as Orchestrator
    participant D as Decoding Engine
    participant M as Model

    U->>O: prompt text
    O->>O: build instruction prefix<br/>(function list + prompt + JSON opener)
    O->>D: generate function name (TrieMatcher)
    D->>M: token-by-token forward passes
    M-->>D: constrained tokens
    D-->>O: function name

    O->>O: append `", "parameters": {` to context
    loop for each parameter
        O->>O: append `"param_name": ` label to context
        O->>D: generate value (NumberGrammar or StringGrammar)
        D->>M: token-by-token forward passes
        M-->>D: constrained tokens
        D-->>O: parameter value
        O->>O: append generated tokens to context
    end

    O->>O: assemble FunctionCallResults
```

### 7.1 Why an instruction prefix is needed

Early in development, the model was only ever shown the bare user prompt —
no list of available functions, no framing at all. It had no way to know
what functions existed or what they did, so function selection was close to
random. The fix was to build an explicit instruction block before generation
begins, listing every available function's name, parameter names/types, and
description, followed by the user's actual request and the opening of the
target JSON object (`{"name": "`). This gives the tiny model something
concrete to reason from instead of guessing blind.

### 7.2 Why context is threaded between stages

Similarly, each generation stage (function name, then each parameter in
turn) is fed the *entire* JSON built so far, not just the original prompt.
Without this, the model generating a parameter value would have no idea
which function it had already committed to, or which parameter slot it was
currently filling in. Threading the growing token sequence through every
stage keeps the model's context consistent with the JSON actually being
constructed.

## 8. Data / Class Relationships

```mermaid
classDiagram
    class FunctionDef {
        +name: str
        +description: str
        +parameters: dict~str, ParameterType~
        +returns: ParameterType
    }
    class ParameterType {
        +type: "number" | "string"
    }
    class PromptEntry {
        +prompt: str
    }
    class FunctionCallResults {
        +prompt: str
        +name: str
        +parameters: dict~str, float | str~
    }

    class TrieMatcher {
        +get_valid_token_ids(prefix) list~int~
        +is_complete(prefix) bool
    }
    class NumberGrammar {
        +get_valid_token_ids(text) list~int~
        +is_complete(text) bool
    }
    class StringGrammar {
        +get_valid_token_ids(text) list~int~
        +is_complete(text) bool
    }

    class Orchestrator {
        +call_function(prompt, functions, model, vocab) FunctionCallResults
    }

    FunctionDef "1" --> "many" ParameterType
    Orchestrator ..> TrieMatcher : uses
    Orchestrator ..> NumberGrammar : uses
    Orchestrator ..> StringGrammar : uses
    Orchestrator ..> FunctionDef : reads
    Orchestrator ..> FunctionCallResults : produces
```

## 9. End-to-End Pipeline (CLI Run)

```mermaid
flowchart LR
    A[Parse CLI args] --> B[Load JSON files]
    B --> C[Validate against schemas<br/>skip invalid entries]
    C --> D[Initialize model + vocabulary<br/>cached after first run]
    D --> E{For each valid prompt}
    E --> F[Orchestrator.call_function]
    F --> G[Collect FunctionCallResults]
    G --> E
    E --> H[Write results JSON to disk]
```

## 10. Key Guarantees and Their Limits

**Guaranteed:**
- Output is always syntactically valid JSON matching the expected schema.
- The function name is always one of the functions actually provided.
- Numbers and strings always conform to JSON's literal grammar (no stray
  characters, unterminated quotes, or malformed numbers).

**Not guaranteed (inherent to using a very small model):**
- That the *correct* function is chosen for a given prompt.
- That the parameter *values* semantically match what the user asked for.

The project's stated requirement was specifically about syntactic
reliability — not spontaneous correctness — so this split is intentional:
grammar owns syntax, the model (aided by the instruction prefix) owns
semantics, and only the former is mechanically enforced.

## 11. Performance Considerations

- **No KV-cache**: every generated token re-runs a full forward pass over the
  whole sequence so far. This is the single biggest cost in the pipeline.
- **Vocabulary caching**: the tokenizer vocabulary is decoded and indexed once
  and cached to disk (`data/cache/cache.json`), avoiding repeated rebuilds.
- **Precomputed "safe token" sets**: for grammar states that admit a very
  large fraction of the vocabulary (e.g. "inside a string"), the set of
  always-valid tokens is computed once per grammar instance rather than
  re-derived character-by-character on every generation step.
- **Iteration caps and dead-end detection**: generation for any single value
  is bounded, and stops immediately if the grammar ever has no valid next
  token, instead of exhausting the iteration budget uselessly.

## 12. Summary

The system treats constrained decoding as a hard boundary: everything that
can be decided deterministically (JSON punctuation, structure, which
characters are legal at a given point) is decided deterministically, and
everything left to the model (which function, which values) is done inside
a fence narrow enough that it cannot escape into invalid syntax. This is
what allows a 0.6B parameter model — far too small to reliably "just get
JSON right" on its own to still produce 100% structurally valid output.
