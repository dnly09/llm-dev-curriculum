"""
Stage 1 — optional completionist run: FULL fine-tune of a small model (not QLoRA).

Uses the same dataset, split, and eval methodology as train_qlora.py so the
resulting loss curves are directly comparable to the 8B QLoRA run. Deliberately
uses plain HF Transformers + TRL (not Unsloth's LoRA-specific kernels) since
there's no adapter to accelerate here — every parameter is trainable.

Model: Qwen2.5-1.5B-Instruct (ungated, right in the guide's 0.5-1.5B full-FT range).

Usage:
    python train_full_ft.py
"""
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ = 2048
OUTPUT_DIR = "outputs_full_ft"

# 1) Load base in bf16 — no quantization, since full FT needs to backprop into
#    every weight cleanly (can't meaningfully full-finetune 4-bit quantized weights)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, attn_implementation="sdpa",
)
model.gradient_checkpointing_enable()

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {n_trainable:,} / {n_total:,} "
      f"({100 * n_trainable / n_total:.1f}%)  <- should be ~100%, unlike QLoRA's ~0.5%")

# 2) Same data, same split, same seed as the QLoRA run — for a fair comparison
ds = load_dataset("mlabonne/FineTome-100k", split="train[:2000]")


def to_text(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["conversations"], tokenize=False, add_generation_prompt=False)}


# FineTome's raw column is "conversations" (ShareGPT-style); Unsloth's
# standardize_sharegpt() isn't used here since we're going plain HF/TRL —
# apply_chat_template on Qwen's tokenizer expects role/content dicts, so
# reshape ShareGPT's {"from": ..., "value": ...} into that format first.
ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}


def reshape(ex):
    return {"conversations": [{"role": ROLE_MAP.get(m["from"], m["from"]), "content": m["value"]}
                               for m in ex["conversations"]]}


ds = ds.map(reshape)
ds = ds.map(to_text)

split = ds.train_test_split(test_size=0.08, seed=3407)   # same seed as train_qlora.py
train_ds, eval_ds = split["train"], split["test"]
print(f"Train: {len(train_ds)} | Eval (held out): {len(eval_ds)}")

# 3) Train — note the much smaller learning rate vs. LoRA's 2e-4.
#    Full FT updates every weight directly, so it's far more sensitive;
#    a LoRA-sized LR here would likely destabilize or catastrophically forget.
trainer = SFTTrainer(
    model=model, processing_class=tokenizer,
    train_dataset=train_ds, eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text", max_length=MAX_SEQ,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,       # effective batch = 8, same as QLoRA run
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=20,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-5,                  # ~10x smaller than the LoRA run's 2e-4
        logging_steps=1,
        optim="adamw_8bit",                  # keeps optimizer state memory down
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        bf16=True,
    ),
)

train_result = trainer.train()
print("\nFinal train loss:", train_result.training_loss)

eval_result = trainer.evaluate()
print("Final eval loss:", eval_result["eval_loss"])

model.save_pretrained("full_ft_model")
tokenizer.save_pretrained("full_ft_model")
print("\nSaved full fine-tuned model to ./full_ft_model")
print("Compare this eval-loss curve against the 8B QLoRA run's — same data, same split.")