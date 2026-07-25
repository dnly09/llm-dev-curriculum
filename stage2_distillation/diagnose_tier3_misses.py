"""
Tier 3 -- diagnose tier3_gkd's false positives (premature function calls on
examples that should have gotten a clarifying question instead).

Same instinct as the Tier 1 miss-by-miss diagnosis and Tier 2's
diagnose_tier2_misses.py overlap check: don't trust the headline F1 number
without checking WHERE the errors are. Working hypothesis (see conversation
log): tier3_gkd's FP=74 (vs. teacher's own FP=59, and baseline_sft's FP=3)
is explained by the teacher's known over-calling bias leaking into training
unfiltered through the on-policy branch -- unlike Tier 1 (explicitly
filtered) and Tier 2 (captured against ground truth, never the teacher's own
free generation). This script tests that hypothesis directly: if tier3_gkd's
FPs substantially overlap with the teacher's FPs, that supports "inherited
bias." If tier3_gkd has many FPs the teacher gets RIGHT, that points to a
training-instability explanation instead (e.g. the no-warmup, cold-start
on-policy dynamics discussed in PROGRESS.md).

Reuses the exact held-out split and prompt construction from
score_call_vs_clarify_tier3.py -- do not change independently.

Usage:
    python diagnose_tier3_misses.py
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

# Only the models needed to test the overlap hypothesis -- teacher (source of
# the suspected bias) and baseline_sft (near-perfect reference, for contrast).
MODELS = {
    "tier3_gkd":    {"path": "./gkd_student_tier3",        "four_bit": False},
    "teacher":      {"path": "Qwen/Qwen2.5-7B-Instruct",   "four_bit": True},
    "baseline_sft": {"path": "./student_baseline_sft",     "four_bit": False},
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
    return examples


def load_model(model_key, cfg):
    print(f"Loading {model_key} from {cfg['path']}...")
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


def get_predictions(model, tok, examples, batch_size=4):
    """Returns a list of bool: True if the model called a function."""
    preds = []
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

        for j in range(len(batch)):
            gen_tokens = out[j][enc["input_ids"].shape[1] :]
            completion = tok.decode(gen_tokens, skip_special_tokens=True).strip()
            parsed = extract_json(completion)
            pred_is_call = parsed is not None and isinstance(parsed, dict) and "name" in parsed
            preds.append(pred_is_call)
    return preds


def main():
    examples = build_held_out_examples()
    print(f"Held-out set: {len(examples)} examples\n")

    all_preds = {}
    for model_key, cfg in MODELS.items():
        model, tok = load_model(model_key, cfg)
        print(f"Generating predictions for {model_key}...")
        all_preds[model_key] = get_predictions(model, tok, examples)
        del model
        torch.cuda.empty_cache()

    # False positives: gt says clarify (gt_is_call=False), model called anyway.
    fp_sets = {}
    for model_key, preds in all_preds.items():
        fp_sets[model_key] = {
            i for i, (ex, pred) in enumerate(zip(examples, preds))
            if not ex["gt_is_call"] and pred
        }
        print(f"{model_key} FP count: {len(fp_sets[model_key])}")

    tier3_fp = fp_sets["tier3_gkd"]
    teacher_fp = fp_sets["teacher"]
    baseline_fp = fp_sets["baseline_sft"]

    overlap_with_teacher = tier3_fp & teacher_fp
    unique_to_tier3 = tier3_fp - teacher_fp
    overlap_with_baseline = tier3_fp & baseline_fp

    print("\n" + "=" * 70)
    print("OVERLAP ANALYSIS: tier3_gkd's false positives")
    print("=" * 70)
    print(f"tier3_gkd total FPs:              {len(tier3_fp)}")
    print(f"  also FP for teacher:             {len(overlap_with_teacher)} "
          f"({100*len(overlap_with_teacher)/max(1,len(tier3_fp)):.1f}%)")
    print(f"  unique to tier3_gkd (teacher got these RIGHT): {len(unique_to_tier3)} "
          f"({100*len(unique_to_tier3)/max(1,len(tier3_fp)):.1f}%)")
    print(f"  also FP for baseline_sft (near-perfect ref):   {len(overlap_with_baseline)}")
    print("=" * 70)

    print(
        "\nInterpretation guide:\n"
        "- High overlap with teacher FPs -> supports 'unfiltered teacher bias leaked\n"
        "  through the on-policy branch' (Tier 1/2 filtering steps had no equivalent here).\n"
        "- High UNIQUE-to-tier3_gkd count (teacher got these right) -> points away from\n"
        "  inherited bias and toward training-instability / cold-start on-policy dynamics\n"
        "  (no SFT warm-start, no-warmup LR schedule) as the primary cause instead.\n"
    )

    print("Sample of unique-to-tier3_gkd false positives (teacher correctly clarified):")
    for i in list(unique_to_tier3)[:5]:
        print(f"  - user_turn: {examples[i]['user_turn'][:150]}...")


if __name__ == "__main__":
    main()