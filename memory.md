# memory.md — project context

## Project
42 Core Curriculum — **call_me_maybe**: LLM function-calling tool. Translates
natural-language prompts into structured function calls using a local 0.6B model
(Qwen3-0.6B via `llm_sdk.Small_LLM_Model`) with **constrained decoding** to
guarantee 100% valid, schema-compliant JSON output.

Input files: `data/input/functions_definition.json`, `data/input/function_calling_tests.json`
Output: `data/output/function_calling_results.json` — one object per prompt:
`{"prompt": str, "name": str, "parameters": {name: value}}`.

## Key decisions
- Decoding approach: **full grammar-based JSON generation** (token-by-token state
  machine over the output schema) — NOT the hybrid json.dumps approach.
- Greedy argmax selection, no sampling.
- No hardcoding: everything derives from the loaded functions_definition.json
  (peer review swaps in different function sets).

## Current state
- Done: CLI parsing (`src/arguement_parser.py`), JSON loading w/ error handling
  (`src/json_loader.py`), Pydantic models (`src/models.py`), input validation
  orchestration in `src/__main__.py`.
- Missing: LLM + constrained-decoding layer, output writer, tests.
- `src/test.py` is scratch code; Makefile expects `pytest test` (so tests should
  live in a `test/` directory).
- Env: macOS (Apple Silicon), uv 0.12.3, `.venv` exists, Makefile has
  `install / run / test-scripts / debug / clean / lint` (flake8 + mypy).
- Venv installed: torch 2.13.0, transformers 5.14.1, llm-sdk importable.
  llm-sdk deps: torch, transformers>=4.40, huggingface-hub. Main deps: numpy, pydantic.

## Phase 0 — model accessibility: VERIFIED (build-mode session)
- `Small_LLM_Model()` initializes OK; `Small_LLM_Model` has NO public `.device`
  attribute (AttributeError) — device is managed internally; don't reference it.
- `get_logits_from_input_ids(ids)` works — returns logits array (vocab ~151,936
  for Qwen3-0.6B). This is the core raw material for constrained decoding.
- `encode()`/`decode()` round-trip confirmed.
- Diagnostic used: `inspect.getsource(Small_LLM_Model.__init__)` and `dir(m)`
  to discover public API.
- First run downloads Qwen3-0.6B (~1.2 GB fp16) from HF Hub to
  `~/.cache/huggingface/hub` (was empty; disk 80 GB free).

## Planned work (Phases)
0. Setup: `make install`, first `uv run python -m src` to download model & sanity-check SDK. — VERIFIED (see above).
1. Infra: `src/vocab.py` (vocab id<->token text, byte-level decode), `src/model.py`
   (lazy singleton wrapper: encode/decode/logits). — NEXT UP.
2. Grammar: `src/constraints.py` — NumberConstraint (`-?\d+(\.\d+)?`), StringConstraint
   (no unescaped quotes/control chars, `"` = terminator), NameConstraint (prefix match).
3. Decoder: `src/decoder.py` — state machine (FIXED / NAME / NUMBER / STRING / END
   nodes), mask logits to allowed token set, argmax, advance state.
4. Orchestration: rework `src/__main__.py` (load -> validate -> model -> per-prompt
   constrained decode -> collect); `src/output.py` writes results (auto-create dir).
5. Tests: replace `src/test.py` with `test/` pytest suite (unit + integration + bad input).
6. Verify: `uv run python -m src`, `pytest test`, `make lint`.

## Gotchas to handle
- Qwen tokenizer is byte-level BPE: decode token strings via bytes; leading-space
  (Ġ) handling matters for numbers/strings.
- Numbers: reject tokens with leading/trailing spaces or stray chars; stop before `,`/`}`.
- Strings: never allow early `"` inside value; cap length (~20 tokens) as safety valve.
- Fallback if NAME node stalls: retry once with rephrased template; else emit flagged entry.

## Session log
- 2026-08-17: reviewed question.md, produced build plan, chose full grammar-based
  constrained decoding, initialized this memory file.
- 2026-08-17 (build session): Phase 0 verified — model loads, logits accessible
  (~151,936 vocab), encode/decode round-trip OK. Note: no public `.device` attr.
  Next: Phase 1 — build `src/vocab.py` + `src/model.py`.