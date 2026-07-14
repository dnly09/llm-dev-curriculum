"""
Tier 3 — prep the Qwen train split into the raw `messages` format
GKDTrainer's DataCollatorForChatML expects.

This intentionally reuses the exact split logic from capture_teacher_logits.py /
filter_teacher_completions.py (same N_EXAMPLES / SPLIT_SEED / TEST_SIZE, same
build_example_messages shape) so Tier 3 trains on the identical 2760-example
train split Tier 1 and Tier 2 used, and any held-out eval stays the same 240
examples score_call_vs_clarify_tier2.py already scored against. Do not change
these constants independently of the Tier 1/2 scripts -- they must match or
the splits silently diverge.

Why raw `messages`, not the pre-flattened "text" column from Stage 1's
train_function_calling.py: GKDConfig forces `dataset_kwargs =
{"skip_prepare_dataset": True}`, so DataCollatorForChatML receives the dataset
untouched and re-derives prompt vs. completion itself by re-applying the chat
template to `messages[:-1]` (prompt) and `messages` (full). A pre-flattened
"text" string has no message boundaries left for it to find.

Why the ORIGINAL ground-truth conversation (system + human + real next turn),
not the Tier-1 teacher-generated completion: same reasoning as Tier 2 design
decision #1 in capture_teacher_logits.py. When lmbda < 1, GKD's off-policy
branch trains on the dataset's own labels each time it doesn't sample an
on-policy rollout -- that should be the clean ground truth, not another
model's approximation of it, or we've quietly reintroduced Tier 1's "SFT on
teacher text" ceiling into part of the training signal.

Output: a HF Dataset saved to disk with a single "messages" column
(system/user/assistant triples), ready for `load_from_disk()` in train_gkd.py.

Usage:
    python prep_gkd_dataset.py --split unfiltered --output ./gkd_train_unfiltered/
    python prep_gkd_dataset.py --split filtered   --output ./gkd_train_filtered/
    python prep_gkd_dataset.py --split unfiltered --output ./gkd_eval/ --held-out
"""

import json
import re
import argparse

from datasets import load_dataset, Dataset

# --- must match capture_teacher_logits.py / filter_teacher_completions.py exactly ---
N_EXAMPLES = 3000
SPLIT_SEED = 3407
TEST_SIZE = 0.08
COMPLETIONS_PATH = "teacher_completions.jsonl"

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)

ROLE_MAP = {"human": "user", "gpt": "assistant"}  # function_call handled specially


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_split():
    """Returns (train_ds, held_out_ds) -- identical to capture_teacher_logits.py's
    build_train_split(), except we also keep the held-out half instead of
    discarding it, since Tier 3 may want it for GKDConfig(eval_dataset=...)."""
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=SPLIT_SEED).select(range(min(N_EXAMPLES, len(ds))))
    split = ds.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
    return split["train"], split["test"]


def get_filtered_indices(train_ds):
    """Verbatim from capture_teacher_logits.py -- reproduces
    filter_teacher_completions.py's pass/fail classification against the
    already-saved teacher_completions.jsonl. Only used with --split filtered."""
    completions = [json.loads(l) for l in open(COMPLETIONS_PATH)]
    assert len(train_ds) == len(
        completions
    ), "ordering assumption broken -- do not proceed"

    keep_idx = []
    for i, (ex, row) in enumerate(zip(train_ds, completions)):
        convo = ex["conversations"]
        expects_fc = len(convo) >= 2 and convo[1]["from"] == "function_call"
        parsed = extract_json(row["teacher_completion"])
        teacher_emitted_json = parsed is not None

        if expects_fc:
            if teacher_emitted_json and isinstance(parsed, dict) and "name" in parsed:
                keep_idx.append(i)
        else:
            if not teacher_emitted_json:
                keep_idx.append(i)
    return keep_idx


def build_example_messages(ex):
    """Verbatim from capture_teacher_logits.py: system (tools) + human + the
    real next turn (function_call or plain assistant reply). Exactly the
    3-message [system, user, assistant] shape DataCollatorForChatML expects --
    messages[:-1] becomes the generation prompt, the full list becomes the
    labeled sequence."""
    tools = json.loads(ex["tools"]) if ex["tools"] else []
    system = SYSTEM_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
    convo = ex["conversations"]
    if not convo or convo[0]["from"] != "human" or len(convo) < 2:
        return None

    messages = [{"role": "system", "content": system}]
    for turn in convo[:2]:  # only human + its immediate response
        role = ROLE_MAP.get(turn["from"])
        if turn["from"] == "function_call":
            messages.append({"role": "assistant", "content": turn["value"]})
        elif role is not None:
            messages.append({"role": role, "content": turn["value"]})
        else:
            return None
    if len(messages) != 3:
        return None
    return messages


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["filtered", "unfiltered"], default="unfiltered")
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--held-out",
        action="store_true",
        help="Build from the 240-example held-out split instead of the 2760-example "
        "train split (e.g. for GKDConfig(eval_dataset=...)). Filtering does not apply "
        "to the held-out split -- it stays untouched, matching how score_call_vs_clarify "
        "scripts use it.",
    )
    args = ap.parse_args()

    train_ds, held_out_ds = build_split()

    if args.held_out:
        src_ds = held_out_ds
        print(f"Held-out split: {len(src_ds)} examples (no filtering applied)")
    elif args.split == "filtered":
        keep_idx = set(get_filtered_indices(train_ds))
        src_ds = train_ds.select(sorted(keep_idx))
        print(f"Filtered train split: {len(src_ds)} examples")
    else:
        src_ds = train_ds
        print(f"Unfiltered train split: {len(src_ds)} examples")

    rows, n_skipped = [], 0
    for ex in src_ds:
        messages = build_example_messages(ex)
        if messages is None:
            n_skipped += 1
            continue
        rows.append({"messages": messages})

    print(f"Built: {len(rows)} | Skipped (malformed): {n_skipped}")

    out_ds = Dataset.from_list(rows)
    out_ds.save_to_disk(args.output)
    print(f"Saved to {args.output}")
    print('Load in train_gkd.py with: datasets.load_from_disk("' + args.output + '")')


if __name__ == "__main__":
    main()