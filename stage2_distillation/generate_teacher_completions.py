"""
Stage 2, Tier 1 — sequence-level KD: teacher generates completions offline.

Idea: teacher generates high-quality completions on the SAME function-calling
prompts (training split only) -> student does plain SFT on those completions
using the existing Stage 1 pipeline. This is tokenizer-agnostic (no logit
alignment needed) and reuses 100% of Stage 1's SFT machinery.

CRITICAL: reproduces the exact filter -> shuffle(seed=3407) -> select(3000) ->
train_test_split(test_size=0.08, seed=3407) split from train_function_calling.py
/ score_function_calling.py, so we only generate on the 2760-example TRAIN half.
The 240-example held-out TEST half must never be touched here -- it's the same
set score_function_calling.py evaluates against, and Stage 2's own eval (student
+ distillation vs. student + plain SFT vs. teacher) depends on it staying clean.

Usage:
    python generate_teacher_completions.py
"""
import json

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
OUTPUT_PATH = "teacher_completions.jsonl"

# --- must match train_function_calling.py / score_function_calling.py exactly ---
N_EXAMPLES = 3000
SPLIT_SEED = 3407
TEST_SIZE = 0.08

MAX_NEW_TOKENS = 150
GEN_BATCH_SIZE = 4          # tune down if you hit OOM; teacher is 4-bit but generation is memory-heavy

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)


def build_train_split():
    """Reproduce train_function_calling.py's split exactly; return only the TRAIN half."""
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=SPLIT_SEED).select(range(min(N_EXAMPLES, len(ds))))
    split = ds.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
    train_ds = split["train"]
    print(f"Reproduced split -> train: {len(train_ds)} | test (held out, untouched): {len(split['test'])}")
    return train_ds


def build_prompt(example):
    """Only the human turn(s) up to the first function_call -- the same prompt shape
    score_function_calling.py builds -- so the teacher is solving the same task."""
    tools = json.loads(example["tools"]) if example["tools"] else []
    system = SYSTEM_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
    convo = example["conversations"]
    if not convo or convo[0]["from"] != "human":
        return None
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": convo[0]["value"]},
    ]


def main():
    print("Loading teacher tokenizer + model (4-bit NF4)...")
    tok = AutoTokenizer.from_pretrained(TEACHER_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"   # required for correct batched generation

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16
    )
    teacher.eval()

    train_ds = build_train_split()

    prompts, kept_examples = [], []
    for ex in train_ds:
        msgs = build_prompt(ex)
        if msgs is None:
            continue
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(text)
        kept_examples.append(ex)

    print(f"Generating completions for {len(prompts)} training-split prompts "
          f"(batch size {GEN_BATCH_SIZE})...")

    results = []
    with open(OUTPUT_PATH, "w") as f:
        for i in range(0, len(prompts), GEN_BATCH_SIZE):
            batch_prompts = prompts[i : i + GEN_BATCH_SIZE]
            batch_examples = kept_examples[i : i + GEN_BATCH_SIZE]

            enc = tok(batch_prompts, return_tensors="pt", padding=True).to(teacher.device)
            with torch.no_grad():
                out = teacher.generate(
                    **enc,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,          # greedy -- deterministic, matches score script's expectations
                    pad_token_id=tok.pad_token_id,
                )

            for j, ex in enumerate(batch_examples):
                gen_tokens = out[j][enc["input_ids"].shape[1] :]
                completion = tok.decode(gen_tokens, skip_special_tokens=True).strip()
                row = {
                    "tools": ex["tools"],
                    "prompt_user_turn": ex["conversations"][0]["value"],
                    "teacher_completion": completion,
                }
                f.write(json.dumps(row) + "\n")
                results.append(row)

            if (i // GEN_BATCH_SIZE) % 10 == 0:
                print(f"  {min(i + GEN_BATCH_SIZE, len(prompts))}/{len(prompts)} done")

    print(f"\nWrote {len(results)} teacher completions to {OUTPUT_PATH}")
    print("Next: point a student SFT run (Stage 1's train_full_ft.py, LR 2e-5 per the "
          "guide) at this file instead of the raw glaive dataset.")


if __name__ == "__main__":
    main()