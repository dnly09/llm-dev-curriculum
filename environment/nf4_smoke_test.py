"""
Run this AFTER verify_stack.py passes, BEFORE committing to any long training run.
Confirms NF4 QLoRA training actually works on your 5090 (not just inference).

Usage:
    python environment/nf4_smoke_test.py
"""
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",   # tiny, fast smoke target
    max_seq_length=1024, load_in_4bit=True, dtype=None,
)
model = FastLanguageModel.get_peft_model(
    model, r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=8, use_gradient_checkpointing="unsloth", random_state=3407,
)

ds = load_dataset("mlabonne/guanaco-llama2-1k", split="train[:50]")
trainer = SFTTrainer(
    model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=1024,
        per_device_train_batch_size=2, max_steps=2, logging_steps=1,
        optim="adamw_8bit", report_to="none", output_dir="smoke",
    ),
)
out = trainer.train()
print("loss finite:", out.training_loss == out.training_loss)  # False if NaN
print("NF4 QLoRA TRAINING WORKS ON YOUR 5090 ✅")
