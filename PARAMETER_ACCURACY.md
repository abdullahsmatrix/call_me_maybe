# How Parameter Accuracy Was Fixed

This document explains how the pipeline went from **5/11 correct parameter
sets to 11/11**, and why each fix was needed. It assumes you've read
`README.md`; it goes deeper on the two modules that do the real work —
the decoding engine and the orchestrator.

The point of this document is the *reasoning*, not the diff. Every fix
below came from measuring something specific, and several of my first
hypotheses were wrong. Those wrong turns are included on purpose, because
knowing what *didn't* explain a bug is most of what makes the real
explanation convincing.

---

## Part 1 — How the two core modules work

### 1.1 The decoding engine (`src/decoder.py`)

An LLM generates text one token at a time. At each step it outputs
**logits** — one score per token in the vocabulary (~150k of them).
Normally you take the highest-scoring token and repeat.

Constrained decoding inserts one step in the middle:

```
logits = model.get_logits_from_input_ids(input_ids)   # 1. what does the model want?
valid  = grammar.get_valid_token_ids(accumulated)     # 2. what is legal right now?
masked = mask_logits(logits, valid)                   # 3. everything illegal -> -inf
next   = argmax(masked)                               # 4. pick the best legal option
```

Because step 3 happens *before* step 4, an illegal token can never be
chosen. The model isn't being trusted to behave — it's being handed a
menu with only legal items on it. That's the whole idea.

`generate_constrained()` is grammar-agnostic. It never knows whether it's
building a function name, a number, or a string. It only needs an object
with two methods:

| method | question it answers |
|---|---|
| `get_valid_token_ids(text_so_far)` | which token ids are legal next? |
| `is_complete(text_so_far)` | is this a valid finished value? |

The four grammars in `src/grammar.py` all implement that contract:
`TrieMatcher` (must match one of a fixed set of strings), `NumberGrammar`
and `IntegerGrammar` (JSON number state machines), and `StringGrammar`
(JSON quoted-string state machine).

**Stopping is harder than it looks.** The loop has four exits:

1. **Dead end** — the grammar allows nothing; nothing valid can follow.
2. **Complete and unextendable** — the value is finished and there is no
   legal continuation (e.g. a string's closing quote was just emitted).
   Checked without a model call, so it costs nothing.
3. **Complete, extendable, but the model wants out** — see §2.1.
4. **Budget exhausted** — `max_iterations`, or the repetition guard
   (§2.2) trips.

### 1.2 The orchestrator (`src/orchestrator.py`)

The orchestrator's job is to produce one complete function call. Its key
design decision is that **the model never generates JSON structure.**
Braces, commas, quotes, and key names are written directly by our code,
because they're 100% predictable — there's no reason to spend a model
call on them or risk getting them wrong. The model is only ever asked to
fill in the parts that actually vary.

A single call is built in three stages, all sharing one growing token
sequence (`context_ids`) so each stage can see what earlier stages
decided:

```
1. Build an instruction prefix: the rules, the list of available
   functions with their parameter names/types, the user's request, and
   the opening of the answer:  Answer: {"function": "

2. Generate the function name, constrained by a TrieMatcher built from
   the actually-available function names. The model literally cannot
   name a function that doesn't exist.

3. Write  ", "arguments": {  ourselves, then fill each parameter in turn.
   Before each one, write its label — e.g.  "encoding":  — so the model
   knows which slot it is filling.
```

How a parameter gets filled depends on its declared type:

| type | mechanism |
|---|---|
| `number` / `integer` | masked constrained decoding via `NumberGrammar` / `IntegerGrammar` |
| `boolean` | masked constrained decoding via `TrieMatcher(['true','false'])` |
| `string` | **candidate scoring** — see §3 |

The final JSON keys (`prompt`, `name`, `parameters`) come from the
`FunctionCallResults` pydantic model, and the file is written with
`json.dump`. This matters: it means the *scaffold* text shown to the
model is free to use different key names than the output, and any
escaping in the output is handled correctly by `json.dump` no matter what
characters a value contains.

---

## Part 2 — Fixing the generation loop

### 2.1 Multi-digit numbers were truncated to one digit

**Symptom:** `12345` came out as `1`; `987.654` came out as `987.6`.

**Cause.** `NumberGrammar.is_complete("1")` correctly returns `True` —
`1` *is* a complete, valid JSON number. The loop treated that as "stop
now". But "this is *a* valid place to stop" and "stop here" are different
claims, and the loop conflated them. The model never got the chance to
emit a second digit.

This is a genuinely hard case, because unlike a string (which ends with
an unambiguous closing quote), **a number has no terminator character.**
Nothing in `123` says "more digits follow".

**Fix.** Don't stop merely because the value *could* end here. Ask what
the model would pick with **no masking at all**:

```python
if grammar.is_complete(accumulated_text):
    raw_top = int(np.argmax(logits))      # the model's true, unconstrained wish
    if raw_top not in valid_tokens:       # it wants a comma/brace -> it's done
        break
```

If its honest preference is still a legal continuation (another digit),
keep going. If it has moved on to something outside the grammar — a
comma, a closing brace — the value is finished. No extra model call is
needed, because those logits were already fetched this iteration.

This fix is generic, which paid off later: it also handles the case where
one function name is a prefix of another (`fn_add` vs `fn_add_numbers`).

### 2.2 The program appeared to freeze

**Symptom:** one prompt took **318 seconds**; the whole run took 418s,
well past the 5-minute budget, with no output the entire time.

**Cause.** Greedy decoding (temperature 0, no repetition penalty) on a
small model can fall into a self-reinforcing loop: once it emits
`product: database; product: production;`, that phrase becomes its own
most likely continuation. `StringGrammar`'s "inside a string" state
legally permits almost the whole vocabulary, so nothing stopped it, and
it never chose to emit a closing quote. It ran the full 60-token budget —
and because there's no KV-cache, every extra token re-runs a full forward
pass over an ever-longer sequence, so cost grows roughly quadratically.

**Fix.** `_has_repetition_loop()` watches the tail of the generated token
ids for a short cycle (period 2–8) repeating three or more times, and
bails out. Period 1 is deliberately excluded so a legitimately repeated
digit inside a number is never cut off.

**Result:** that prompt went 318s → 40s; the full run 418s → 157s.

This treats the symptom, not the cause — the output was garbage either
way. But garbage in 40s beats garbage in 318s, and §3 removed the cause.

> **A fix I tried and reverted.** I also tried *proactively* blocking the
> model from ever completing a repeat (forcing it to its second choice).
> Measured, it was worse: the model just switched to a near-repeat
> (`production database` / `production db`) that dodged the detector, so
> generation ran the *full* budget instead of being cut short — and it
> broke a prompt that previously worked. Reverted. The lesson: verify a
> plausible-sounding fix actually helps before keeping it.

---

## Part 3 — The real fix for string parameters

After Part 2 the output was *fast and syntactically perfect* but often
semantically wrong. Scored against `data/input/function_calling_corrections.json`:
**11/11 function names, 5/11 parameter sets.** Every failure was a string
parameter. Numbers, integers and booleans were already 100%.

Typical failures:

```
expected: {'query': 'SELECT * FROM users',        'database': 'production'}
got:      {'query': 'user:SELECT * FROM users; production database; ...',
           'database': 'user:production database'}

expected: {'path': 'C:\\Users\\john\\config.ini', 'encoding': 'latin-1'}
got:      {'path': '/Users/john/config.ini',      'encoding': 'encoding=utf-16'}
```

Three distinct diseases: leaked junk prefixes (`user:`, `encoding=`),
trailing hallucination, and plain wrong content.

### 3.1 The insight: stop generating, start choosing

The breakthrough came from reading another student's solution
(`AlexHysel/42-Call-Me-Maybe`) and noticing a fundamentally different
approach to string values.

Look at the expected answers above. **Every correct value is a literal
substring of the prompt.** `SELECT * FROM users`, `production`,
`latin-1`, `C:\Users\john\config.ini` — all of them appear verbatim in
the request. The model was never supposed to *write* anything. It was
supposed to *copy* something.

So the question changes from an open one to a closed one:

> ~~"What characters should this value contain?"~~
> **"Which span of the prompt does this value correspond to?"**

The second question has a small, finite answer set. So we build it
(`_extract_prompt_candidates`):

- every bare word and every quoted phrase in the prompt, and
- everything after the prompt's last colon — which captures
  `Format template: <the whole rest>` style requests, where the entire
  remainder is one value.

Now hallucination and repetition loops are **impossible by construction**.
The model cannot emit `user:` or invent `utf-16`, because neither is on
the menu. It can only pick a real span.

Two side benefits fell out for free:

- **Backslash escaping stopped mattering.** The model had been mangling
  `C:\Users\john\config.ini` into `/Users/john/config.ini` because it
  couldn't reliably produce JSON escape sequences. Now we copy the span
  and `json.dump` escapes it. Nothing to get wrong.
- **Two earlier patch-fixes became dead code and were deleted** — a
  function that stripped leaked `user:` / `encoding=` prefixes, and one
  that pasted a quoted span over a noisy value. Both existed only to
  clean up artifacts this approach prevents outright.

This lifted parameters from **6/11 → 9/11**.

> **What I did NOT copy.** That repo also hardcodes keyword→regex
> mappings (`"vowel"` → `[aeiouAEIOU]`) for its own test set. That's the
> exact "hardcode solutions based on the provided examples" the subject
> forbids — it wouldn't survive a different function set. The reusable
> idea is the candidate mechanism; the hardcoded table is not.

### 3.2 Choosing badly: greedy picks the first token, not the best answer

Two cases still failed, and the first was subtle. For
`Read the file at /home/user/data.json with utf-8 encoding`, the `path`
slot got `file`.

The correct candidate *was* on the menu. So why lose? Because the trie
walk is **greedy on the first token**. Printing the allowed tokens with
their logits:

```
step 0, slot "path":
    15.17  'file'      <- wins immediately, and completes in one token
    13.15  'the'
    ...             no '/'-starting token even reaches the top 8
```

Two problems compound here:

1. **Echo.** The context immediately before is
   `...fn_read_file", "arguments": {"path": "`. The token `file` appears
   right there in the function name, so it's cheap to repeat.
2. **Greed.** The decision is made on token 1 alone. `file` completes in
   one token and wins there, even though `/home/user/data.json` is far
   more likely *taken as a whole*.

**Fix:** stop walking the trie greedily and instead **score each complete
candidate**, then pick the best. Measured on that exact slot:

| scoring rule | picks |
|---|---|
| total log-probability | ❌ `file` (−6.01) |
| **mean log-probability per token** | ✅ `/home/user/data.json` (−2.95) |

Why *mean* and not *total*: total log-prob sums a negative number per
token, so it mechanically favours short candidates — the same bias that
broke greedy. Dividing by length makes candidates of different lengths
comparable. (This is the standard length-normalisation used in beam
search.)

### 3.3 Scoring an unterminated prefix is not scoring the value

Mean log-prob fixed the path, but immediately broke the templates:
`Hello {user}'s profile!` came back as just `Hello`.

Same disease in a new place. `Hello` is a perfectly likely thing to say —
*if you're allowed to keep talking*. Scoring it in isolation never
charges it for the thing it would have to do next: **close the string**.

**Fix:** score each candidate together with its closing quote — i.e.
score the complete, terminated JSON value `"Hello"` versus
`"Hello {user}'s profile!"`, not the bare prefixes. Now stopping early
has to justify itself, and `Hello` no longer can, because the model
plainly doesn't want a quote there.

This is the same insight as §2.1 (a completion must be *wanted*, not just
*legal*) — expressed as scoring instead of as a stopping rule.

### 3.4 Slots competing for the same span

Last failure, and the most interesting. `path` and `encoding` came back
**swapped**. Measuring both slots against all candidates explains it
exactly:

| candidate | score for `path` | score for `encoding` |
|---|---|---|
| `utf-8` | **−4.45** | **−2.44** ← near-certain |
| `/home/user/data.json` | −4.97 | −7.58 |

`utf-8` narrowly outbids the path for the `path` slot (−4.45 vs −4.97).
Filling slots left to right, `path` takes `utf-8` and locks it — even
though `encoding` wanted it far more urgently (−2.44), and the path
candidate had nowhere else to go.

This is an **assignment problem**, and greedy left-to-right is simply the
wrong algorithm for it. The local decision is right; the global outcome
is wrong.

**Fix (`_assign_candidates_globally`):** collect every slot's scores for
every candidate, then assign the most confident `(slot, candidate)` pairs
first, skipping any slot or candidate already taken. Here the strongest
pair overall is `(encoding, utf-8)` at −2.44 — assigned first — after
which `path` takes `/home/user/data.json`. Correct.

This is only possible because string values are **selected rather than
generated**: there's no token stream to unwind, so the assignment can be
revised after the fact. The one approximation is that each slot's scores
were measured in a context threaded with the earlier provisional picks.

### 3.5 One more echo: the parameter name as its own value

`"template": "template"`. Models happily repeat the key as the value.
Since a value is essentially never literally the parameter's own name,
that candidate is dropped from the slot's menu. Generic, one line.

---

## Part 4 — The function-name bug (and a wrong diagnosis)

One prompt kept selecting the wrong function, across *many* unrelated
experiments:

```
Format template: Say "hello" to {name}   ->   fn_read_file   (wrong)
Format template: Hello {user}'s profile! ->   fn_format_template (right)
```

I initially concluded this was an irreducible quirk of a 0.6B model and
said so. **That was wrong**, and the way it was wrong is instructive.

Dumping the model's raw, unconstrained top-5 at the first generation step
showed something strange — for the working prompt it was confident
(`'fn'` at 33.44), but for the failing one its preferences were
`'user'`, `'name'`, `'string'`, `'john'`, `'k'`. Not *a wrong function* —
**not a function name at all.** It wasn't choosing badly; it didn't think
it was naming a function.

Those look like values for a *placeholder*. And the prompt ends with the
placeholder `{name}` — while our scaffold ended with:

```
Answer: {"name": "
```

The model conflated the two and set about filling in `{name}`. One
substitution proved it:

| prompt ending | result |
|---|---|
| `...to {name}` | ❌ `fn_read_file` |
| `...to {xyz}` | ✅ `fn_format_template` |
| `...to {user}` | ✅ `fn_format_template` |

**Fix:** rename the keys in the scaffold the model is *shown* to
`{"function": ...,"arguments": {...}}`. The emitted JSON is unaffected,
because output keys come from `FunctionCallResults`, not from this text.

**A related fix that did NOT solve it.** I also found that
`f'User: "{prompt}"'` didn't escape quotes, so this prompt rendered as
`User: "Format template: Say "hello" to {name}"` — genuinely broken
nested quoting. I fixed it (`_escape_for_display`) and re-measured: the
logits barely moved. It's a real defect worth fixing, but it was **not**
the cause. Confirming a fix actually moves the number is the difference
between fixing a bug and just changing code.

---

## Part 5 — Results and honest limits

Scored against `data/input/function_calling_corrections.json`:

| stage | names | parameters |
|---|---|---|
| before this work | 11/11 | 5/11 |
| after prefix-cleanup patches | 11/11 | 6/11 |
| after candidate substrings (§3.1) | 10/11 | 9/11 |
| after scaffold rename + scoring + assignment | **11/11** | **11/11** |

Runtime ~180s for 11 prompts — inside the subject's 5-minute budget, but
notably slower than greedy, because scoring N candidates costs roughly
`sum(len(candidate_tokens) + 1)` forward passes per slot instead of one
walk. That is the price paid for the accuracy.

**Verified not to be overfitted.** On a completely unseen function set
using JSON-Schema-shaped parameters and `str`/`float`/`int` type aliases,
with prompts never seen before, all 3/3 were correct — including
correctly splitting a quoted span and a trailing word across two
different string slots.

### What is guaranteed vs. what is not

**Guaranteed, mechanically:**
- Output is valid JSON matching the required schema.
- The function name is always one of the provided functions (enforced by
  the trie mask — a name that isn't in the list cannot be produced, no
  matter what the model prefers).
- Numbers/integers/booleans are always well-formed literals.
- String values are always literal spans of the request.

**Not guaranteed:**
- That the *right* function is chosen.
- That the *right* span is chosen for each slot.
- **The core assumption of §3:** that a string value is always a literal
  substring of the prompt. This holds for every case in the subject's
  examples, but a request needing a *transformed* value (e.g. uppercased,
  or a value merely implied rather than written) would not be covered.
  `StringGrammar` free-form generation is kept as a fallback for when no
  candidates can be extracted at all.

### The general lesson

Nearly every bug here was the same shape in different clothes:

> **A locally-correct decision that is globally wrong.**

Stopping at `1` is locally valid JSON. Picking `file` is the best first
token. Choosing `Hello` is a likely thing to say. Giving `utf-8` to
`path` is the best bid for that slot. Every one of those is defensible in
isolation — and wrong once you look one step wider. The fixes all take
the same form: widen the frame before committing (check what the model
wants *next*; score the *whole* candidate; score it *terminated*; assign
*all* slots together).

And the measurement discipline mattered more than the cleverness. Three
plausible fixes in this document were tested and rejected — proactive
repeat-blocking, deleting the few-shot examples, quote-escaping as the
misselection cure — and one confident conclusion ("unfixable model
quirk") was simply wrong. Printing the actual logits is what settled it
every time.
