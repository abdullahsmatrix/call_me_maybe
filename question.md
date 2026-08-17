So, heres the project I am trying to work on:

V.1 Summary
In this project, you will create a function calling tool that translates natural language
prompts into structured function calls. Given a question like "What is the sum of 40 and
2?", your solution should not return 42, but instead provide:
• The function name: fn_add_numbers
• The arguments: {"a": 40, "b": 2}
Your implementation must use constrained decoding to guarantee 100% valid JSON
output, ensuring near-perfect reliability even with a small 0.6B parameter model.


Your solution will process two input files located in the data/input/ directory:
• function_calling_tests.json: contains a JSON array of natural language prompts
that your system must process.
Example: function_calling_tests.json

[
{
"prompt": "What is the sum of 2 and 3?"
},
{
"prompt": "What is the sum of 265 and 345?"
},
{
"prompt": "Greet shrek"
},
{
"prompt": "Greet john"
},
{
"prompt": "Reverse the string 'hello'"
},
...
]

• functions_definition.json: contains the available functions your system can
call. Each function includes:
◦ Function name
◦ Argument names and types
◦ Return type
◦ Description
Example: functions_definition.json

These examples establish the expected complexity level. However,
your solution will be tested with different prompts and function
sets. You must implement proper JSON error handling for input files,
as they may contain invalid JSON or be missing entirely.


V.3 LLM Interaction
V.3.1 The LLM SDK
Attached to this project, you’ll find a wrapper class Small_LLM_Model in the llm_sdk
package that you can use to interact with the LLM.
The SDK provides several essential methods:
• get_logits_from_input_ids(input_ids: List[int]) -> List[float]
Takes a list of token IDs and returns the logits produced by the LLM model.
• get_path_to_vocab_file() -> str
Returns the path to the vocabulary file containing the correspondence between
token IDs and tokens.
• encode(text: str) -> Tensor
Encodes a text string into a tensor of token IDs using the model’s tokenizer.
• decode(token_ids: List[int]) -> str (optional)
Optionally decodes a list of token IDs back into a text string.
V.3.2 The Generation Pipeline
The LLM generation process follows these steps:

1. Prompt: Your natural language question
Example: "What is the sum of 2 and 3?"
2. Tokenization: The text is broken into subword units (tokens). Unlike simple word
splitting, tokenizers often include leading spaces, handle punctuation, and split
words into smaller components using algorithms such as BPE or SentencePiece.
Example (realistic): ["What", "Gis ˙ ", "Gthe ˙ ", "Gsum ˙ ", "Gof ˙ ", "G2˙ ", "Gand ˙ ", "G3˙ ", "?"]
Note: The symbol "˙G", indicates a preceding space; real tokenizers preserve such
details to reconstruct text accurately.
3. Input IDs: Tokens are converted to numerical IDs the model understands.
Example (illustrative): [892, 318, 262, 4771, 286, 16, 290, 17, 30]
4. LLM Processing: The model processes these numbers through its neural network.
5. Logits: The model outputs probability scores for each possible next token.
Example: token_5: 0.001, token_42: 0.85, token_100: 0.02, ...
6. Token Selection: The next token is chosen based on these probabilities, usually
the one with the highest score.
At this stage, techniques like constrained decoding can be applied to restrict the
token choices and ensure outputs follow a specific structure, such as generating
100% valid JSON.

Important: This process repeats token-by-token. Each generated token is added to the
prompt, and steps 2-6 repeat until the complete response is generated.
Simplified view:
Prompt -> Tokenization -> Input IDs -> LLM -> Logits -> Next Token Selection
V.3.3 Understanding Constrained Decoding
Language models generate text one token at a time. At each step, the model produces
a probability distribution (logits) over all possible next tokens. Normally, you would
sample from this distribution or pick the highest probability token.
Constrained decoding intervenes in this process by modifying the logits before token
selection:
1. The model produces logits for all possible tokens.
2. You identify which tokens would maintain both a valid JSON structure and compliance with the expected schema.
3. You set logits for invalid tokens (those breaking the schema or structure) to negative
infinity.
4. You sample only from the remaining valid tokens.

In this project, constrained decoding must not only ensure syntactically valid JSON
but also enforce a specific schema. For instance, if a field is constrained to a number
in functions_definition.json, the decoder restricts token selection to values satisfying
either an integer or a float, preserving both JSON validity and schema compliance. This
guarantees that every generated token maintains both structural and semantic validity,
enforcing the required schema. As a result, the produced JSON is 100% retrievable and
can always be parsed without errors.
Your solution must NOT rely on the model spontaneously producing
correct JSON from a prompt. Prompting the model with function
definitions and hoping for structured output is not reliable, and
it is not the skill we expect you to develop here.
Think about how you can use the vocabulary JSON file to map between
tokens and their string representations. This is crucial for
determining which tokens are valid at each generation step.

V.4 Output File Format
Your program will produce a single JSON file: data/output/function_calling_results.json.
For each prompt, add a JSON object to this file. Each object in the array must contain
exactly the following keys:
• prompt (string): The original natural-language request
• name (string): The name of the function to call
• parameters (object): All required arguments with the correct types

V.4.1 Example Output
[
{
"prompt": "What is the sum of 2 and 3?",
"name": "fn_add_numbers",
"parameters": {"a": 2.0, "b": 3.0}
},
{
"prompt": "Reverse the string 'hello'",
"name": "fn_reverse_string",
"parameters": {"s": "hello"}
}
]
V.4.2 Validation Rules
• The file must be valid JSON (no trailing commas, no comments)
• Keys and types must match the schema in functions_definition.json exactly
• No extra keys or prose are allowed anywhere in the output
• All required arguments must be present
• Argument types must match the function definition (number, string, boolean, etc.)

The given input files may change during the peer review. Do not
hardcode solutions based on the provided examples.

V.5 Performance and Reliability
Your implementation should achieve:
• Near-perfect accuracy: 90%+ correct function selection and argument extraction
• 100% valid JSON: Every output must be parseable and schema-compliant
• Reasonable speed: Process all test prompts in under 5 minutes on standard
hardware
• Robust error handling: Gracefully handle malformed inputs, missing files, and
edge cases
The Qwen3-0.6B model has only 500 million parameters, yet with proper
constrained decoding, it can achieve reliability comparable to much
larger models. This demonstrates the power of structural guidance
over raw model size.

V.6 Testing Your Implementation
To verify your solution works correctly:
1. Ensure input files are in the data/input/ directory
2. Run: uv run python -m src [–functions_definition <function_definition_file>]
[–input <input_file>] [–output <output_file>]
3. Check that output/function_calling_results.json is created
4. Validate the JSON structure and content
5. Verify function names and argument types match the definitions
Test with various edge cases: empty strings, large numbers, special
characters, wrong types, ambiguous prompts, and functions with
multiple parameters.