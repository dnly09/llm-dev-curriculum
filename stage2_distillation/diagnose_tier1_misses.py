"""
Stage 2, Tier 1 diagnostic -- inspect the distilled student's exact_args
MISSES on the held-out set, to distinguish a real capability gap from a
formatting-convention mismatch (student learned the teacher's argument-filling
style, which may differ from the dataset's own ground-truth conventions even
when both are reasonable answers).

Usage:
    python diagnose_tier1_misses.py
"""
import json
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

N_EXAMPLES = 3000
SPLIT_SEED = 3407
TEST_SIZE = 0.08
MAX_NEW_TOKENS = 150
STUDENT_PATH = "./student_seqkd_full"

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_held_out_fc_examples():
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=SPLIT_SEED).select(range(min(N_EXAMPLES, len(ds))))
    split = ds.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
    held_out = split["test"]

    fc_examples = []
    for ex in held_out:
        convo = ex["conversations"]
        if len(convo) >= 2 and convo[0]["from"] == "human" and convo[1]["from"] == "function_call":
            try:
                gt = json.loads(convo[1]["value"])
            except json.JSONDecodeError:
                continue
            if isinstance(gt, dict) and "name" in gt:
                fc_examples.append({
                    "tools": ex["tools"],
                    "user_turn": convo[0]["value"],
                    "gt_name": gt["name"],
                    "gt_args": gt.get("arguments", {}),
                })
    return fc_examples


def main():
    fc_examples = build_held_out_fc_examples()
    print(f"Scoring {len(fc_examples)} held-out function-call examples with {STUDENT_PATH}...")

    tok = AutoTokenizer.from_pretrained(STUDENT_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(STUDENT_PATH, dtype="bfloat16").to("cuda")
    model.eval()

    misses = []
    batch_size = 4
    for i in range(0, len(fc_examples), batch_size):
        batch = fc_examples[i : i + batch_size]
        prompts = []
        for ex in batch:
            tools = json.loads(ex["tools"]) if ex["tools"] else []
            system = SYSTEM_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": ex["user_turn"]},
            ]
            prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

        enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id
            )

        for j, ex in enumerate(batch):
            gen_tokens = out[j][enc["input_ids"].shape[1] :]
            completion = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            parsed = extract_json(completion)

            correct_name = parsed is not None and isinstance(parsed, dict) and parsed.get("name") == ex["gt_name"]
            exact_args = correct_name and parsed.get("arguments", {}) == ex["gt_args"]

            if not exact_args:
                misses.append({
                    "user_turn": ex["user_turn"],
                    "gt_name": ex["gt_name"],
                    "gt_args": ex["gt_args"],
                    "pred_name": parsed.get("name") if isinstance(parsed, dict) else None,
                    "pred_args": parsed.get("arguments") if isinstance(parsed, dict) else None,
                    "name_matched": correct_name,
                })

    print(f"\n{len(misses)} misses out of {len(fc_examples)}\n")
    print("=" * 80)
    for m in misses:
        print(f"USER: {m['user_turn'][:120]}")
        print(f"  name matched: {m['name_matched']}")
        print(f"  GT   name={m['gt_name']!r}  args={m['gt_args']}")
        print(f"  PRED name={m['pred_name']!r}  args={m['pred_args']}")
        print("-" * 80)


if __name__ == "__main__":
    main()