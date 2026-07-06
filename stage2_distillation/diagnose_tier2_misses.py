"""
Stage 2, Tier 2 diagnostic -- inspect exact_args MISSES on the held-out set,
and check overlap against baseline_sft's misses (same method Tier 1 used:
diagnose_tier1_misses.py) to distinguish a real capability gap from a
formatting-convention mismatch.

Set STUDENT_PATH below to whichever checkpoint you want to inspect:
    STUDENT_PATH = "./tier2_student_unfiltered"   # <- default, strongest candidate
    STUDENT_PATH = "./tier2_student_filtered"
    STUDENT_PATH = "./student_seqkd_full"          # tier1, for direct comparison

Usage:
    python diagnose_tier2_misses.py
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

STUDENT_PATH = "./tier2_student_unfiltered"
BASELINE_PATH = "./student_baseline_sft"

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


def score_and_collect_misses(model_path, fc_examples, batch_size=4):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="bfloat16").to("cuda")
    model.eval()

    misses = {}  # keyed by index into fc_examples, so overlap comparison is exact
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
            idx = i + j
            gen_tokens = out[j][enc["input_ids"].shape[1] :]
            completion = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            parsed = extract_json(completion)

            correct_name = parsed is not None and isinstance(parsed, dict) and parsed.get("name") == ex["gt_name"]
            exact_args = correct_name and parsed.get("arguments", {}) == ex["gt_args"]

            if not exact_args:
                misses[idx] = {
                    "user_turn": ex["user_turn"],
                    "gt_name": ex["gt_name"],
                    "gt_args": ex["gt_args"],
                    "pred_name": parsed.get("name") if isinstance(parsed, dict) else None,
                    "pred_args": parsed.get("arguments") if isinstance(parsed, dict) else None,
                    "name_matched": correct_name,
                }

    del model
    torch.cuda.empty_cache()
    return misses


def main():
    fc_examples = build_held_out_fc_examples()
    n = len(fc_examples)

    print(f"Scoring {n} held-out function-call examples with {STUDENT_PATH}...")
    student_misses = score_and_collect_misses(STUDENT_PATH, fc_examples)

    print(f"Scoring {n} held-out function-call examples with {BASELINE_PATH} (for overlap check)...")
    baseline_misses = score_and_collect_misses(BASELINE_PATH, fc_examples)

    student_idx = set(student_misses.keys())
    baseline_idx = set(baseline_misses.keys())
    shared = student_idx & baseline_idx
    student_only = student_idx - baseline_idx

    print(f"\n{STUDENT_PATH}: {len(student_idx)}/{n} misses")
    print(f"{BASELINE_PATH}: {len(baseline_idx)}/{n} misses")
    print(f"Shared misses (same example, likely shared 0.5B-scale limitation or scoring artifact): {len(shared)}")
    print(f"Misses unique to {STUDENT_PATH} (worth inspecting closely): {len(student_only)}")

    # sub-check on shared misses: did both models predict the SAME (wrong) values,
    # which is the strongest signal of a shared limitation rather than two different errors
    identical_wrong = sum(
        1 for idx in shared
        if student_misses[idx]["pred_name"] == baseline_misses[idx]["pred_name"]
        and student_misses[idx]["pred_args"] == baseline_misses[idx]["pred_args"]
    )
    print(f"  of which both models predicted IDENTICAL (wrong) values: {identical_wrong}")

    print("\n" + "=" * 80)
    print(f"MISSES UNIQUE TO {STUDENT_PATH} (not also missed by baseline_sft):")
    print("=" * 80)
    for idx in sorted(student_only):
        m = student_misses[idx]
        print(f"USER: {m['user_turn'][:120]}")
        print(f"  name matched: {m['name_matched']}")
        print(f"  GT   name={m['gt_name']!r}  args={m['gt_args']}")
        print(f"  PRED name={m['pred_name']!r}  args={m['pred_args']}")
        print("-" * 80)


if __name__ == "__main__":
    main()