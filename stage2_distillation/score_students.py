"""
Stage 2, Tier 1 eval -- head-to-head scoring on the untouched 240-example held-out set.

Compares four models on the SAME held-out prompts (the test half of the split
that generate_teacher_completions.py / filter / both training scripts never
touched):
    1. base_student   -- Qwen2.5-0.5B-Instruct, no training  (floor)
    2. baseline_sft    -- plain SFT on raw glaive ground truth (the bar to beat)
    3. tier1_distilled -- SFT on filtered teacher completions (the payoff)
    4. teacher          -- Qwen2.5-7B-Instruct, 4-bit           (ceiling)

Scoring is 3-level, only on held-out examples where ground truth actually
expects a function call (consistent with Stage 1's score_function_calling.py
methodology):
    - valid_json:     completion parses to a JSON object with a "name" key
    - correct_name:   valid_json AND name matches ground truth exactly
    - exact_args:      correct_name AND arguments dict matches ground truth
                        exactly (order-independent key/value match)

Usage:
    python score_students.py
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


def build_held_out_split():
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
    print(f"Held-out set: {len(held_out)} total | {len(fc_examples)} with function_call ground truth "
          "(only these are scored, matching Stage 1 methodology)")
    return fc_examples


def load_model(model_key, cfg):
    print(f"\nLoading {model_key} from {cfg['path']}...")
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    if cfg["four_bit"]:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], quantization_config=bnb_config, torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], dtype="bfloat16").to("cuda")
    model.eval()
    return model, tok


def score_model(model, tok, fc_examples, batch_size=4):
    valid_json_n = correct_name_n = exact_args_n = 0

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

            if parsed is not None and isinstance(parsed, dict) and "name" in parsed:
                valid_json_n += 1
                if parsed["name"] == ex["gt_name"]:
                    correct_name_n += 1
                    if parsed.get("arguments", {}) == ex["gt_args"]:
                        exact_args_n += 1

    n = len(fc_examples)
    return {
        "valid_json": (valid_json_n, n),
        "correct_name": (correct_name_n, n),
        "exact_args": (exact_args_n, n),
    }


def main():
    fc_examples = build_held_out_split()

    results = {}
    for model_key, cfg in MODELS.items():
        model, tok = load_model(model_key, cfg)
        print(f"Scoring {model_key} on {len(fc_examples)} held-out function-call examples...")
        results[model_key] = score_model(model, tok, fc_examples)
        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'Model':<18} {'valid_json':<16} {'correct_name':<16} {'exact_args':<16}")
    print("=" * 70)
    for model_key, res in results.items():
        vj = f"{res['valid_json'][0]}/{res['valid_json'][1]} ({100*res['valid_json'][0]/res['valid_json'][1]:.1f}%)"
        cn = f"{res['correct_name'][0]}/{res['correct_name'][1]} ({100*res['correct_name'][0]/res['correct_name'][1]:.1f}%)"
        ea = f"{res['exact_args'][0]}/{res['exact_args'][1]} ({100*res['exact_args'][0]/res['exact_args'][1]:.1f}%)"
        print(f"{model_key:<18} {vj:<16} {cn:<16} {ea:<16}")
    print("=" * 70)
    print("\nSuccess criterion (guide sec 6): tier1_distilled should beat baseline_sft on exact_args,")
    print("closing a meaningful fraction of the gap toward teacher.")


if __name__ == "__main__":
    main()