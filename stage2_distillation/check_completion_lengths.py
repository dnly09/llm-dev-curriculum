"""
Tier 3 — measure completion token lengths on the prepped GKD dataset to
right-size max_new_tokens for the real run, instead of guessing.

The 256 used in the smoke-test config was inherited from the Stage 2 guide's
generic GKD example, not measured against this task. Function-calling
completions here are short JSON objects (compare Tier 2's MAX_SEQ_LEN=768 for
the WHOLE sequence, prompt included) -- 256 tokens of headroom for just the
completion is almost certainly far more than needed, and directly inflates
both wall-clock time and per-step VRAM during on-policy generation.

Usage:
    python check_completion_lengths.py
"""

from datasets import load_from_disk
from transformers import AutoTokenizer

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    ds = load_from_disk("./gkd_train_unfiltered/")

    lengths = []
    for ex in ds:
        messages = ex["messages"]  # [system, user, assistant]
        assistant_content = messages[2]["content"]
        # Rough measure: tokenize just the completion turn's content on its own.
        # Not byte-identical to how the collator will tokenize it in-context
        # (chat template adds role wrapper tokens), but close enough to size
        # max_new_tokens -- we want a ceiling, not an exact count.
        n_tokens = len(tokenizer(assistant_content, add_special_tokens=False)["input_ids"])
        lengths.append(n_tokens)

    lengths.sort()
    n = len(lengths)
    def pct(p):
        return lengths[min(int(n * p), n - 1)]

    print(f"n = {n}")
    print(f"min: {lengths[0]}  |  max: {lengths[-1]}")
    print(f"p50: {pct(0.50)}  |  p90: {pct(0.90)}  |  p95: {pct(0.95)}  |  p99: {pct(0.99)}")
    print(f"mean: {sum(lengths)/n:.1f}")


if __name__ == "__main__":
    main()