"""
Stage 2, Tier 1 — student SFT on filtered teacher completions (sequence-level KD).

The student is full-fine-tuned (not QLoRA -- it's small enough, and the guide's
own VRAM table + Stage 1 findings put small-student full-FT at LR 2e-5) on
teacher_completions_filtered.jsonl: the teacher's completions, minus the
hallucinated-premature-call examples filtered out by filter_teacher_completions.py.

Both behaviors are represented in training:
    - expected_behavior == "function_call"      -> assistant turn is the JSON call
    - expected_behavior == "clarify_or_decline"  -> assistant turn is the teacher's
                                                     plain-text clarifying question/decline

Internal train/eval split here (92/8) is ONLY for loss-curve monitoring during
training. It is NOT the held-out set used for final scoring -- that's the
240-example split score_function_calling.py already owns, reproduced by the
same filter -> shuffle(seed=3407) -> select(3000) -> train_test_split(seed=3407)
recipe, and this script never touches it.

Usage:
    python train_student_seqkd.py
"""
import json

from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
INPUT_PATH = "teacher_completions_filtered.jsonl"
OUTPUT_DIR = "outputs_student_seqkd"
MAX_SEQ = 2048

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)


def load_filtered_rows():
    rows = []
    with open(INPUT_PATH) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def to_text(row, tokenizer):
    tools = json.loads(row["tools"]) if row["tools"] else []
    system = SYSTEM_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": row["prompt_user_turn"]},
        {"role": "assistant", "content": row["teacher_completion"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def main():
    print("Loading student tokenizer + model (bf16, full precision -- it's small)...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    model = AutoModelForCausalLM.from_pretrained(STUDENT_ID, dtype="bfloat16").to("cuda")

    print("Loading filtered teacher completions...")
    rows = load_filtered_rows()
    n_call = sum(1 for r in rows if r["expected_behavior"] == "function_call")
    n_clarify = sum(1 for r in rows if r["expected_behavior"] == "clarify_or_decline")
    print(f"Loaded {len(rows)} rows ({n_call} function_call, {n_clarify} clarify_or_decline)")

    texts = [to_text(r, tokenizer) for r in rows]
    ds = Dataset.from_dict({"text": texts})

    split = ds.train_test_split(test_size=0.08, seed=3407)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)} | Internal eval (loss monitoring only, NOT the held-out scoring set): {len(eval_ds)}")

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
            num_train_epochs=2,          # small full-FT student; a bit more signal per example than QLoRA needs
            learning_rate=2e-5,          # per guide's full-FT default for small students
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

    model.save_pretrained("student_seqkd_full")
    tokenizer.save_pretrained("student_seqkd_full")
    print("\nSaved student to ./student_seqkd_full")
    print("Next: score with score_function_calling.py (point BASE_MODEL/ADAPTER_PATH at this "
          "student, or add a no-adapter full-model-path branch) against the SAME 240-example "
          "held-out set, and compare to a plain-SFT student trained directly on raw glaive data.")


if __name__ == "__main__":
    main()