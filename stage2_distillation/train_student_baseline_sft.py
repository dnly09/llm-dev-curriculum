"""
Stage 2, Tier 1 eval baseline — plain SFT student (NO teacher involved).

This is "the bar distillation must clear" from the guide's §6 eval table.
Same student architecture, same reproduced train split, same LR/epoch count
as train_student_seqkd.py -- the only variable that changes is where the
training targets come from: here, straight from the dataset's own ground
truth (both function_call turns and plain-text clarifying/declining turns),
not the teacher's generations.

Uses the exact same 2760-example TRAIN split (never touches the 240-example
held-out set score_function_calling.py owns).

Usage:
    python train_student_baseline_sft.py
"""
import json

from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
OUTPUT_DIR = "outputs_student_baseline"
MAX_SEQ = 2048

N_EXAMPLES = 3000
SPLIT_SEED = 3407
TEST_SIZE = 0.08

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)


def build_train_split():
    """Same reproduction recipe as generate_teacher_completions.py / filter script."""
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=SPLIT_SEED).select(range(min(N_EXAMPLES, len(ds))))
    split = ds.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
    train_ds = split["train"]
    print(f"Reproduced split -> train: {len(train_ds)} | test (held out, untouched): {len(split['test'])}")
    return train_ds


def ground_truth_target(example):
    """Return the raw ground-truth assistant turn: either a function_call
    (re-serialized as {"name": ..., "arguments": ...} to match the same
    output format the student is expected to produce) or a plain gpt turn."""
    convo = example["conversations"]
    if len(convo) < 2 or convo[0]["from"] != "human":
        return None
    second = convo[1]
    if second["from"] == "function_call":
        try:
            parsed = json.loads(second["value"])
        except json.JSONDecodeError:
            return None
        if not (isinstance(parsed, dict) and "name" in parsed):
            return None
        return json.dumps(parsed)
    elif second["from"] == "gpt":
        return second["value"]
    return None


def to_text(example, target, tokenizer):
    tools = json.loads(example["tools"]) if example["tools"] else []
    system = SYSTEM_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": example["conversations"][0]["value"]},
        {"role": "assistant", "content": target},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    print("Loading student tokenizer + model (bf16, full precision)...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_ID, dtype="bfloat16").to("cuda")

    train_split = build_train_split()

    texts = []
    n_call, n_clarify = 0, 0
    for ex in train_split:
        target = ground_truth_target(ex)
        if target is None:
            continue
        texts.append(to_text(ex, target, tokenizer))
        if ex["conversations"][1]["from"] == "function_call":
            n_call += 1
        else:
            n_clarify += 1

    print(f"Built {len(texts)} ground-truth training examples "
          f"({n_call} function_call, {n_clarify} clarify_or_decline)")

    ds = Dataset.from_dict({"text": texts})
    split = ds.train_test_split(test_size=0.08, seed=3407)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)} | Internal eval (loss monitoring only): {len(eval_ds)}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text",
            max_length=MAX_SEQ,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=4,
            eval_strategy="steps",
            eval_steps=30,
            warmup_steps=10,
            num_train_epochs=2,          # match Tier 1 student's epoch count for fair comparison
            learning_rate=2e-5,          # match Tier 1 student's LR for fair comparison
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    train_result = trainer.train()
    print("\nFinal train loss:", train_result.training_loss)

    eval_result = trainer.evaluate()
    print("Final eval loss:", eval_result["eval_loss"])

    model.save_pretrained("student_baseline_sft")
    tokenizer.save_pretrained("student_baseline_sft")
    print("\nSaved baseline student to ./student_baseline_sft")
    print("Next: score both ./student_seqkd_full and ./student_baseline_sft against the "
          "SAME 240-example held-out set to see whether distillation actually beat plain SFT.")


if __name__ == "__main__":
    main()