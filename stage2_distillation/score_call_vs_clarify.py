"""
Stage 2, Tier 1 eval -- "call vs clarify" binary classification.

Separate from the argument exact_args metric: this only asks whether the
model correctly decides TO call a function or not, ignoring argument content
entirely. Ground truth is unambiguous (convo[1]["from"] == "function_call"),
so this sidesteps every formatting-convention issue that made exact_args hard
to interpret cleanly (case, singular/plural, empty-placeholder keys, dataset
internal inconsistency).

Non-trivial: filter_teacher_completions.py already showed the 7B TEACHER
itself gets this wrong ~46% of the time on examples where a call isn't yet
warranted (calls prematurely instead of asking for missing required args).
So a clean win here, if it appears, reflects a real judgment capability, not
a saturated or ambiguous task.

Reuses the four checkpoints already trained -- no new training required.
Scores the FULL 240-example held-out set (not just the function_call subset),
since this metric doesn't need argument-level ground truth.

Usage:
    python score_call_vs_clarify.py
"""
import json
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

N_EXAMPLES = 3000
SPLIT_SEED = 3407
TEST_SIZE = 0.08
MAX_NEW_TOKENS = 150

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)

MODELS = {
    "base_student":    {"path": "Qwen/Qwen2.5-0.5B-Instruct", "four_bit": False},
    "baseline_sft":    {"path": "./student_baseline_sft",     "four_bit": False},
    "tier1_distilled": {"path": "./student_seqkd_full",       "four_bit": False},
    "teacher":         {"path": "Qwen/Qwen2.5-7B-Instruct",   "four_bit": True},
}


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def build_held_out_examples():
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=SPLIT_SEED).select(range(min(N_EXAMPLES, len(ds))))
    split = ds.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
    held_out = split["test"]

    examples = []
    for ex in held_out:
        convo = ex["conversations"]
        if len(convo) < 2 or convo[0]["from"] != "human":
            continue
        examples.append({
            "tools": ex["tools"],
            "user_turn": convo[0]["value"],
            "gt_is_call": convo[1]["from"] == "function_call",
        })
    n_call = sum(1 for e in examples if e["gt_is_call"])
    print(f"Held-out set: {len(examples)} total ({n_call} call / {len(examples) - n_call} clarify)")
    return examples


def load_model(model_key, cfg):
    print(f"\nLoading {model_key} from {cfg['path']}...")
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if cfg["four_bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], quantization_config=bnb_config, torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], dtype="bfloat16").to("cuda")
    model.eval()
    return model, tok


def score_model(model, tok, examples, batch_size=4):
    tp = fp = fn = tn = 0

    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
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
            pred_is_call = parsed is not None and isinstance(parsed, dict) and "name" in parsed

            gt = ex["gt_is_call"]
            if gt and pred_is_call:
                tp += 1
            elif not gt and pred_is_call:
                fp += 1
            elif gt and not pred_is_call:
                fn += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy,
    }


def main():
    examples = build_held_out_examples()

    results = {}
    for model_key, cfg in MODELS.items():
        model, tok = load_model(model_key, cfg)
        print(f"Scoring {model_key} on {len(examples)} held-out examples (call vs clarify)...")
        results[model_key] = score_model(model, tok, examples)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print(f"{'Model':<18} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Accuracy':<12} {'TP/FP/FN/TN':<20}")
    print("=" * 90)
    for model_key, res in results.items():
        confusion = f"{res['tp']}/{res['fp']}/{res['fn']}/{res['tn']}"
        print(
            f"{model_key:<18} {res['precision']*100:>6.1f}%     {res['recall']*100:>6.1f}%     "
            f"{res['f1']*100:>6.1f}%     {res['accuracy']*100:>6.1f}%     {confusion:<20}"
        )
    print("=" * 90)
    print("\nPositive class = 'called a function'. High precision = few false-positive premature")
    print("calls (the failure mode filter_teacher_completions.py caught in the teacher itself).")
    print("High recall = didn't miss cases that genuinely warranted a call.")


if __name__ == "__main__":
    main()