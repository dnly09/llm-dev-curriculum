"""
Stage 2, Tier 1 — filter teacher_completions.jsonl before training the student.

Diagnosis from check_teacher_completions_v2.py + manual review: where ground
truth expects a function call, the teacher is 100% valid. But where ground
truth does NOT yet expect one (the dataset's multi-turn convention is:
clarifying question first, function call only once required arguments are
known), the teacher frequently skips the clarification and calls the function
immediately with HALLUCINATED arguments -- invented password lengths, invented
meeting content, invented review text the user never provided, or a garbled
mix of text-reply + a JSON call with empty-string placeholders.

Training the student on those examples would teach it to confabulate missing
arguments instead of asking for them -- worse than simply not calling a
function. This script keeps only the two behaviors we actually want the
student to learn:
    1. ground truth expects a call, teacher produced valid call-shaped JSON
    2. ground truth does NOT expect a call, teacher correctly emitted no JSON
       (i.e. asked a clarifying question / declined, matching the dataset's
       own convention)
and drops everything else.

Usage:
    python filter_teacher_completions.py
"""
import json
import re

from datasets import load_dataset

INPUT_PATH = "teacher_completions.jsonl"
OUTPUT_PATH = "teacher_completions_filtered.jsonl"
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
    train_ds = build_train_split()
    completions = [json.loads(l) for l in open(INPUT_PATH)]
    assert len(train_ds) == len(completions), "ordering assumption broken -- do not proceed"

    kept, dropped_hallucinated = [], []

    for ex, row in zip(train_ds, completions):
        convo = ex["conversations"]
        expects_fc = len(convo) >= 2 and convo[1]["from"] == "function_call"
        parsed = extract_json(row["teacher_completion"])
        teacher_emitted_json = parsed is not None

        if expects_fc:
            if teacher_emitted_json and isinstance(parsed, dict) and "name" in parsed:
                row["expected_behavior"] = "function_call"
                kept.append(row)
            # else: teacher failed a case it should have called -- drop (rare; check_v2 showed 0 of these)
        else:
            if not teacher_emitted_json:
                row["expected_behavior"] = "clarify_or_decline"
                kept.append(row)
            else:
                dropped_hallucinated.append(row)

    with open(OUTPUT_PATH, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")

    n_call = sum(1 for r in kept if r["expected_behavior"] == "function_call")
    n_clarify = sum(1 for r in kept if r["expected_behavior"] == "clarify_or_decline")

    print(f"Input:  {len(completions)} completions")
    print(f"Kept:   {len(kept)}  ({n_call} function_call + {n_clarify} clarify_or_decline)")
    print(f"Dropped (hallucinated premature call): {len(dropped_hallucinated)}")
    print(f"\nWrote filtered set to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()