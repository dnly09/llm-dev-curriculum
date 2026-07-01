"""
Stage 1, Track B — the real QLoRA fine-tune (Unsloth + TRL).

First run: a small public instruction slice (mlabonne/FineTome-100k) just to
prove the pipeline end-to-end before switching to a domain-specific dataset.

Usage:
    python train_qlora.py
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ = 2048
OUTPUT_DIR = "outputs"

# 1) Load base in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
)

# 2) Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                       # 8/16/32 typical; higher = more capacity + VRAM
    lora_alpha=16,              # common heuristic: alpha == r
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",   # big VRAM saver on 24 GB
    random_state=3407,
)

# 3) Data -> chat template -> "text" column, with a held-out eval split
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
ds = load_dataset("mlabonne/FineTome-100k", split="train[:2000]")
ds = standardize_sharegpt(ds)


def fmt(ex):
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False,
                     add_generation_prompt=False) for c in ex["conversations"]]}


ds = ds.map(fmt, batched=True)

# held out ~8% (≈160 examples) for eval loss tracking — see guide §4
split = ds.train_test_split(test_size=0.08, seed=3407)
train_ds, eval_ds = split["train"], split["test"]
print(f"Train: {len(train_ds)} | Eval (held out): {len(eval_ds)}")

# 4) Train
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=train_ds, eval_dataset=eval_ds,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAX_SEQ,
        per_device_train_batch_size=2,      # effective batch = 2 * 4 = 8
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        eval_strategy="steps",
        eval_steps=20,
        warmup_steps=5,
        # max_steps=60,                       # first run: keep small to prove it works
        num_train_epochs=1,               # switch to this for a fuller real run
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",                 # 8-bit optimizer saves VRAM
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

# 5) Save adapters (small, portable) — merge/GGUF export happens in a later script
model.save_pretrained("lora_finetome")
tokenizer.save_pretrained("lora_finetome")
print("\nSaved LoRA adapters to ./lora_finetome")
print("Next: run eval/compare_base_vs_finetuned.py for a qualitative side-by-side")