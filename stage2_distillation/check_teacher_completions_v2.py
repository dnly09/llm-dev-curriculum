"""
Stage 2, Tier 1 — refined sanity check: separates teacher_completions.jsonl by
whether the GROUND TRUTH for that prompt actually expects a function call.

The naive check (check_teacher_completions.py) counts any non-JSON completion
as a failure, but not every training-split prompt has a matching tool -- some
correctly warrant a plain-text decline (e.g. "order me a pizza" with no pizza
function available). This script re-derives ground truth from the same
deterministic split and reports valid_json rate ONLY on the subset where a
function call was actually expected, which is the number that's actually
comparable to score_function_calling.py's eval methodology.

Usage:
    python check_teacher_completions_v2.py
"""
import json
import re

from datasets import load_dataset

COMPLETIONS_PATH = "teacher_completions.jsonl"
N_EXAMPLES = 3000
SPLIT_SEED = 3407
TEST_SIZE = 0.08


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_train_split():
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=SPLIT_SEED).select(range(min(N_EXAMPLES, len(ds))))
    split = ds.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
    return split["train"]


def main():
    print("Reproducing train split to recover ground truth (same seed as generation)...")
    train_ds = build_train_split()

    completions = []
    with open(COMPLETIONS_PATH) as f:
        for line in f:
            completions.append(json.loads(line))

    assert len(train_ds) == len(completions), (
        f"Length mismatch: train_ds={len(train_ds)} completions={len(completions)} "
        "-- ordering assumption broken, don't trust the pairing below."
    )

    fc_expected_total = 0
    fc_expected_valid_json = 0
    no_fc_expected_total = 0
    no_fc_expected_teacher_still_emitted_json = 0

    for ex, row in zip(train_ds, completions):
        convo = ex["conversations"]
        expects_fc = len(convo) >= 2 and convo[1]["from"] == "function_call"
        parsed = extract_json(row["teacher_completion"])

        if expects_fc:
            fc_expected_total += 1
            if parsed is not None and isinstance(parsed, dict) and "name" in parsed:
                fc_expected_valid_json += 1
        else:
            no_fc_expected_total += 1
            if parsed is not None:
                no_fc_expected_teacher_still_emitted_json += 1

    print(f"\nTotal examples: {len(completions)}")
    print(f"\n--- Where ground truth EXPECTS a function call ({fc_expected_total} examples) ---")
    if fc_expected_total:
        pct = 100 * fc_expected_valid_json / fc_expected_total
        print(f"Teacher emitted valid function-call JSON: {fc_expected_valid_json}/{fc_expected_total} ({pct:.1f}%)")
        print("^ this is the number comparable to score_function_calling.py's valid_json/correct_call metric")

    print(f"\n--- Where ground truth does NOT expect a function call ({no_fc_expected_total} examples) ---")
    if no_fc_expected_total:
        pct = 100 * no_fc_expected_teacher_still_emitted_json / no_fc_expected_total
        print(f"Teacher emitted JSON anyway (potential over-triggering): "
              f"{no_fc_expected_teacher_still_emitted_json}/{no_fc_expected_total} ({pct:.1f}%)")
        print("^ low is good here -- means teacher correctly declines when no tool fits")


if __name__ == "__main__":
    main()