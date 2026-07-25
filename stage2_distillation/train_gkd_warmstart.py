"""
Tier 3 (warm-start variant) -- online on-policy GKD initialized from
tier2_unfiltered (98.4% F1) instead of the raw base Qwen2.5-0.5B-Instruct.

Why: GKD's own literature treats on-policy distillation as a REFINEMENT step
on top of an already-SFT'd student, not a replacement for SFT. The cold-start
full run (train_gkd.py, RUN_MODE="full") trained straight from the base
instruct model with lmbda=0.5 mixed on/off-policy from step 1 -- meaning half
its steps scored the model's own noisy, not-yet-task-competent generations.
Result: tier3_gkd underperformed every other method, including the raw
teacher (76.6% F1 vs. teacher's 80.4%), with FP=74 exceeding even the
teacher's own FP=59 -- see diagnose_tier3_misses.py for the overlap analysis
that should inform whether this warm-start hypothesis is the right target
before spending compute on this run.

Also picks up the create_scheduler fix in gkd_trainer_truncated.py, so
warmup_steps actually applies this time (unconfirmed in the cold-start run).

Same three-stage RUN_MODE pattern as train_gkd.py. Defaults to "medium" --
run that first even though the base config is already validated, since the
warm-start init + fixed scheduler are both new elements worth a cheap
(~13 min) sanity check before committing to another ~2hr full run.

IMPORTANT: OUTPUT_DIR is separate from both prior runs (outputs_gkd /
outputs_gkd_full) so auto-resume can't accidentally pick up an unrelated
checkpoint.

Usage:
    python train_gkd_warmstart.py
"""

import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.experimental.gkd import GKDConfig

from gkd_trainer_truncated import GKDTrainerTruncated

STUDENT_INIT_PATH = "./tier2_student_unfiltered"  # warm-start, not the raw HF id
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"

MAX_LENGTH = 768
OUTPUT_DIR = "outputs_gkd_warmstart"

# --- RUN MODE: "smoke" | "medium" | "full" ---
RUN_MODE = "medium"  # <-- start here again; new config, cheap to re-validate

MEDIUM_N_EXAMPLES = 300
MAX_NEW_TOKENS = 96


def main():
    print(f"Run mode: {RUN_MODE} (warm-start from {STUDENT_INIT_PATH})")

    print("Loading student tokenizer + model (warm-start checkpoint)...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_INIT_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    real_vocab_size = max(len(tokenizer.get_vocab()), max(tokenizer.get_vocab().values()) + 1)
    print(f"Real tokenizer vocab size (truncation target): {real_vocab_size}")

    student = AutoModelForCausalLM.from_pretrained(STUDENT_INIT_PATH, dtype=torch.bfloat16)

    print("Preparing 4-bit teacher config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    teacher_kwargs = {
        "quantization_config": bnb_config,
        "dtype": "bfloat16",
        "device_map": "auto",
    }

    print("Loading training split...")
    train_ds = load_from_disk("./gkd_train_unfiltered/")
    print(f"Full train split: {len(train_ds)} examples")

    if RUN_MODE == "smoke":
        lmbda = 1.0
        max_new_tokens = 32
        run_kwargs = {"max_steps": 5}
        ds = train_ds
    elif RUN_MODE == "medium":
        lmbda = 0.5
        max_new_tokens = MAX_NEW_TOKENS
        run_kwargs = {"num_train_epochs": 1}
        ds = train_ds.shuffle(seed=3407).select(range(MEDIUM_N_EXAMPLES))
        print(f"Medium run subset: {len(ds)} examples")
    elif RUN_MODE == "full":
        lmbda = 0.5
        max_new_tokens = MAX_NEW_TOKENS
        run_kwargs = {"num_train_epochs": 1}
        ds = train_ds
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    cfg = GKDConfig(
        output_dir=OUTPUT_DIR,
        max_length=MAX_LENGTH,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,  # confirmed safe over a full ~2hr run already
        learning_rate=2e-5,
        lmbda=lmbda,
        beta=0.5,
        temperature=2.0,
        max_new_tokens=max_new_tokens,
        teacher_model_init_kwargs=teacher_kwargs,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=3407,
        report_to="none",
        warmup_steps=30,  # now backed by GKDTrainerTruncated.create_scheduler's fix
        save_strategy="steps",
        save_steps=10,
        save_total_limit=3,
        **run_kwargs,
    )

    print("Constructing GKDTrainerTruncated (this loads the teacher)...")
    trainer = GKDTrainerTruncated(
        model=student,
        teacher_model=TEACHER_ID,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        vocab_size=real_vocab_size,
    )

    has_checkpoint = os.path.isdir(OUTPUT_DIR) and any(
        d.startswith("checkpoint-") for d in os.listdir(OUTPUT_DIR)
    )
    if has_checkpoint:
        print(f"Found existing checkpoint(s) in {OUTPUT_DIR} -- resuming.")

    print(f"Starting '{RUN_MODE}' warm-start training run...")
    train_result = trainer.train(resume_from_checkpoint=has_checkpoint)
    print("\nTraining loss:", train_result.training_loss)
    print("loss finite:", train_result.training_loss == train_result.training_loss)
    print("train_runtime (s):", train_result.metrics.get("train_runtime"))

    # Quick check that the scheduler fix actually worked this time -- step 1's LR
    # should be well below 2e-05 if warmup is ramping, not equal to it.
    first_lr = trainer.state.log_history[0].get("learning_rate")
    print(f"\nStep 1 learning_rate: {first_lr} (should be << 2e-05 if warmup is working)")

    if RUN_MODE == "full":
        student.save_pretrained("gkd_student_tier3_warmstart")
        tokenizer.save_pretrained("gkd_student_tier3_warmstart")
        print("\nSaved to ./gkd_student_tier3_warmstart")
    else:
        print(f"\n'{RUN_MODE}' run complete. Check train_runtime and step-1 LR above.")


if __name__ == "__main__":
    main()