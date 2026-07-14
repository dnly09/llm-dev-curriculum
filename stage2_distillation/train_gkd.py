"""
Tier 3 -- online on-policy KD via GKDTrainer (truncated for the Qwen
vocab-padding mismatch). Student generates with probability `lmbda` each
step; teacher scores those rollouts (or the ground-truth labels, off-policy)
via generalized JSD.

Three staged run modes -- run them in order, don't skip to "full":
  "smoke"  -- 5 steps, lmbda=1.0 (always generates), max_new_tokens=32.
              Validates the pipeline end-to-end. DONE (see conversation log):
              loss finite ~0.2-0.29, grad_norm 3-4.25, no crash.
  "medium" -- ~300-example subset, REAL settings (lmbda=0.5,
              max_new_tokens=96), full pass over the subset (~12-15 min
              estimated). Validates sustained stability under real-length
              generation before committing to an unattended multi-hour run --
              this project's WSL2/Hyper-V GPU passthrough previously crashed
              (HYPERVISOR_ERROR) under a different but related generation
              misconfiguration; treat sustained-load stability as unproven
              until this stage passes clean.
  "full"   -- all 2760 examples, num_train_epochs=1. Only run this after
              "medium" has completed without incident and you've sanity
              checked the per-step timing against the medium run's actual
              (not estimated) throughput.

max_new_tokens=96 chosen from check_completion_lengths.py's measured
distribution on this dataset (n=2760, p50=27, p90=46, p95=51, p99=59,
max=81) -- covers the full distribution plus chat-template wrapper-token
margin. The original 256 was an unmeasured guess from the generic Stage 2
guide example and was oversized for this task by ~2.7x.

Checkpointing: save_strategy="steps" with a small save_steps and
save_total_limit, plus resume_from_checkpoint=True (auto-detects the latest
checkpoint in output_dir), so a crash mid-"full"-run costs minutes of
progress, not hours. This was NOT enabled during the smoke test (5 steps,
no checkpoint needed) but is required before "medium"/"full".

Prereqs:
    - ./gkd_train_unfiltered/  (from prep_gkd_dataset.py --split unfiltered)
    - gkd_trainer_truncated.py in the same directory (or on PYTHONPATH)
    - .venv-distill active (trl 1.7.0, TRL_EXPERIMENTAL_SILENCE=1 already
      set in the venv activate script per README_stage2_v2.md)
    - A second WSL2 terminal running `watch -n 1 nvidia-smi` during "medium"
      and "full" -- non-negotiable after the earlier crash, until we have a
      track record of stable multi-hour runs on this hardware.

Usage:
    python train_gkd.py
"""

import os

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl.experimental.gkd import GKDConfig

from gkd_trainer_truncated import GKDTrainerTruncated

STUDENT_ID = "Qwen/Qwen2.5-0.5B-Instruct"
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"

MAX_LENGTH = 768  # matches capture_teacher_logits.py's MAX_SEQ_LEN for consistency
OUTPUT_DIR = "outputs_gkd_full" # medium results saved out to "outputs_gkd"

# --- RUN MODE: "smoke" | "medium" | "full" ---
RUN_MODE = "full"  # <-- smoke and medium already passed

MEDIUM_N_EXAMPLES = 300
MAX_NEW_TOKENS = 96  # measured from check_completion_lengths.py (p99=59, max=81) + margin


def main():
    print(f"Run mode: {RUN_MODE}")

    print("Loading student tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # gotcha #1, same calc as capture_teacher_logits.py: real vocab (~151665)
    # vs padded lm_head (151936 for the 0.5B student, 152064 for the 7B teacher)
    real_vocab_size = max(len(tokenizer.get_vocab()), max(tokenizer.get_vocab().values()) + 1)
    print(f"Real tokenizer vocab size (truncation target): {real_vocab_size}")

    # Full fine-tune, not LoRA -- per the Stage 2 carry-forward lesson, small
    # (0.5-1.5B) students are full-trained; GKDTrainer defaults peft_config=None.
    student = AutoModelForCausalLM.from_pretrained(STUDENT_ID, dtype=torch.bfloat16)

    print("Preparing 4-bit teacher config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    teacher_kwargs = {
        "quantization_config": bnb_config,
        "dtype": "bfloat16",  # REQUIRED key -- GKDConfig.__init__ indexes this
        # unconditionally (teacher_model_init_kwargs["dtype"]), no .get()
        # fallback, so omitting it is a KeyError, not a default.
        "device_map": "auto",
    }

    print("Loading training split...")
    train_ds = load_from_disk("./gkd_train_unfiltered/")
    print(f"Full train split: {len(train_ds)} examples")

    if RUN_MODE == "smoke":
        # Kept for reference / re-running if needed -- already validated.
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
        max_length=MAX_LENGTH,  # NOT max_seq_length -- Stage 1/2 lesson
        per_device_train_batch_size=1,  # generation is memory-heavy; keep small
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,  # CRASH POST-MORTEM: GKDTrainer ties this flag
        # to generation's KV cache too -- see generation_kwargs in gkd_trainer.py:
        # "use_cache": False if args.gradient_checkpointing else True. With it True,
        # every on-policy generation step re-ran a full uncached forward pass per new
        # token (O(n^2), large repeated allocations), which maxed VRAM/GPU-3D and
        # triggered the WSL2 HYPERVISOR_ERROR crash. Student is only 0.5B, so its own
        # backward-pass memory is small even uncheckpointed.
        learning_rate=2e-5,  # full-FT LR for a small full-trained student (Stage 2 lesson)
        lmbda=lmbda,
        beta=0.5,  # 0=forward KL, 1=reverse KL, between=generalized JSD
        temperature=2.0,
        max_new_tokens=max_new_tokens,
        teacher_model_init_kwargs=teacher_kwargs,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        seed=3407,
        report_to="none",
        # Checkpointing -- required for "medium"/"full", harmless for "smoke".
        # 300 examples / (batch 1 * grad_accum 8) = ~37 optimizer steps for "medium";
        # 2760 examples -> ~345 steps for "full". save_steps=10 gives frequent-enough
        # resume points on "medium" (~4 checkpoints) without excessive I/O; revisit
        # for "full" once medium's actual per-step timing is known.
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

    # Auto-resume from the latest checkpoint in OUTPUT_DIR if one exists (e.g.
    # after a crash mid-"full"-run). No-op (False effectively) on a fresh
    # output_dir with no checkpoints yet.
    has_checkpoint = os.path.isdir(OUTPUT_DIR) and any(
        d.startswith("checkpoint-") for d in os.listdir(OUTPUT_DIR)
    )
    if has_checkpoint:
        print(f"Found existing checkpoint(s) in {OUTPUT_DIR} -- resuming.")

    print(f"Starting '{RUN_MODE}' training run...")
    train_result = trainer.train(resume_from_checkpoint=has_checkpoint)
    print("\nTraining loss:", train_result.training_loss)
    print("loss finite:", train_result.training_loss == train_result.training_loss)  # False if NaN
    print("train_runtime (s):", train_result.metrics.get("train_runtime"))

    if RUN_MODE == "full":
        student.save_pretrained("gkd_student_tier3")
        tokenizer.save_pretrained("gkd_student_tier3")
        print("\nSaved to ./gkd_student_tier3")
        print("Next: score_call_vs_clarify_tier3.py against the held-out 240-example split")
    else:
        print(f"\n'{RUN_MODE}' run complete. Check train_runtime above, then decide next stage.")


if __name__ == "__main__":
    main()