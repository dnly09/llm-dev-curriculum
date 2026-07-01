"""
Stage 1 — function-calling eval: base vs fine-tuned, scored by parsing JSON output
against the ground-truth function call, not by reading vibes.

Held-out set: examples from hiyouga/glaive-function-calling-v2-sharegpt where the
FIRST assistant turn is a function_call -- i.e. a single-turn "given this request
and these tools, call the right function with the right arguments" test. Uses the
same filter/shuffle/split as train_function_calling.py so these examples were NOT
seen during training.

Three levels of correctness, scored per example:
    1. valid_json      -- did the output parse as JSON at all?
    2. correct_call     -- valid JSON AND "name" matches AND "arguments" is a dict?
    3. exact_match       -- correct_call AND "arguments" exactly equals the expected dict?

Usage:
    python score_function_calling.py
"""
import json
import re

from unsloth import FastLanguageModel
from datasets import load_dataset

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
ADAPTER_PATH = "../unsloth/lora_function_calling"
MAX_SEQ = 2048
N_EXAMPLES = 3000        # must match train_function_calling.py for identical split
N_EVAL_EXAMPLES = 40     # how many held-out function-call examples to score
MAX_NEW_TOKENS = 150

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)


def build_eval_set():
    """Same filter/shuffle/split as training, then keep only examples whose
    first assistant turn is a function_call -- these become (prompt, expected) pairs."""
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=3407).select(range(min(N_EXAMPLES, len(ds))))
    split = ds.train_test_split(test_size=0.08, seed=3407)   # same seed as training
    test_ds = split["test"]

    examples = []
    for ex in test_ds:
        convo = ex["conversations"]
        if len(convo) >= 2 and convo[0]["from"] == "human" and convo[1]["from"] == "function_call":
            tools = json.loads(ex["tools"])
            system = SYSTEM_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
            prompt_messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": convo[0]["value"]},
            ]
            expected = json.loads(convo[1]["value"])
            examples.append({"messages": prompt_messages, "expected": expected})
        if len(examples) >= N_EVAL_EXAMPLES:
            break
    return examples


def extract_json(text):
    """Model output may include stray whitespace/markdown fences despite instructions;
    grab the first {...} block and try to parse it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate(model, tokenizer, messages):
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    prompt_decoded = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    return full[len(prompt_decoded):].strip()


def score(examples, outputs):
    valid_json = correct_call = exact_match = 0
    details = []
    for ex, raw_output in zip(examples, outputs):
        parsed = extract_json(raw_output)
        is_valid = parsed is not None
        is_correct_call = (is_valid and isinstance(parsed, dict)
                            and parsed.get("name") == ex["expected"].get("name")
                            and isinstance(parsed.get("arguments"), dict))
        is_exact = is_correct_call and parsed.get("arguments") == ex["expected"].get("arguments")

        valid_json += is_valid
        correct_call += is_correct_call
        exact_match += is_exact
        details.append((is_valid, is_correct_call, is_exact, raw_output, ex["expected"]))

    n = len(examples)
    return {
        "valid_json_rate": valid_json / n,
        "correct_call_rate": correct_call / n,
        "exact_match_rate": exact_match / n,
        "n": n,
    }, details


def main():
    print("Building held-out eval set (single-turn function-call examples)...")
    examples = build_eval_set()
    print(f"Eval set size: {len(examples)}")

    print("\nLoading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(base_model)

    print("Generating base model outputs...")
    base_outputs = [generate(base_model, tokenizer, ex["messages"]) for ex in examples]

    import torch, gc
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nLoading fine-tuned (LoRA) model...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    )
    FastLanguageModel.for_inference(ft_model)

    print("Generating fine-tuned model outputs...")
    ft_outputs = [generate(ft_model, ft_tokenizer, ex["messages"]) for ex in examples]

    base_scores, base_details = score(examples, base_outputs)
    ft_scores, ft_details = score(examples, ft_outputs)

    print("\n" + "=" * 70)
    print(f"{'Metric':<25}{'Base':>15}{'Fine-tuned':>15}")
    print("-" * 70)
    for key, label in [("valid_json_rate", "Valid JSON"),
                        ("correct_call_rate", "Correct name+structure"),
                        ("exact_match_rate", "Exact argument match")]:
        print(f"{label:<25}{base_scores[key]*100:>14.1f}%{ft_scores[key]*100:>14.1f}%")
    print("=" * 70)
    print(f"(n={base_scores['n']} held-out examples, not seen during training)")

    print("\nFirst 3 examples in detail:")
    for i in range(min(3, len(examples))):
        print(f"\n--- Example {i+1} ---")
        print(f"Expected: {examples[i]['expected']}")
        print(f"Base:       {base_outputs[i][:150]}")
        print(f"Fine-tuned: {ft_outputs[i][:150]}")


if __name__ == "__main__":
    main()