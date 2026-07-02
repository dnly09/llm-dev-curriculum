"""
Stage 2, Tier 1 — sanity check on teacher_completions.jsonl before training the
student on it. Reuses the same extract_json logic pattern as score_function_calling.py
so the numbers here are directly comparable to eventual student eval numbers.

Usage:
    python check_teacher_completions.py
"""
import json
import re

INPUT_PATH = "teacher_completions.jsonl"
N_PREVIEW = 5


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def main():
    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"Loaded {len(rows)} teacher completions\n")

    print(f"--- First {N_PREVIEW} completions (read these for sanity) ---")
    for row in rows[:N_PREVIEW]:
        print(f"USER: {row['prompt_user_turn'][:100]}")
        print(f"TEACHER: {row['teacher_completion'][:200]}")
        print()

    valid_json_count = 0
    has_name_and_args = 0
    for row in rows:
        parsed = extract_json(row["teacher_completion"])
        if parsed is not None:
            valid_json_count += 1
            if isinstance(parsed, dict) and "name" in parsed and isinstance(parsed.get("arguments"), dict):
                has_name_and_args += 1

    n = len(rows)
    print("--- Aggregate quality check ---")
    print(f"valid_json:        {valid_json_count}/{n}  ({100*valid_json_count/n:.1f}%)")
    print(f"correct_call_shape: {has_name_and_args}/{n}  ({100*has_name_and_args/n:.1f}%)")
    print()
    print("If correct_call_shape is well below ~90%, consider: shorter MAX_NEW_TOKENS")
    print("(teacher may be rambling past the JSON), or check for markdown-fence wrapping")
    print("despite the system prompt's instructions not to use it.")


if __name__ == "__main__":
    main()