"""
Stage 1, Track A — full training loop, raw PyTorch.

Attaches LoRALinear (see lora_linear.py) to a real model's attention projections
and hand-implements the five things a real trainer does:
    1. Mixed precision      (torch.autocast bf16)
    2. Gradient accumulation (simulate a bigger batch on limited VRAM)
    3. Activation checkpointing (trade compute for memory)
    4. Optimizer + LR schedule (AdamW on LoRA params only, cosine w/ warmup)
    5. Checkpoint/resume     (save+reload model/optimizer/scheduler/step)

Model: Qwen2.5-0.5B-Instruct (small, ungated, real attention layers).
Dataset: a 200-row slice of guanaco-llama2-1k (same one from the Phase 0 smoke test).

Usage:
    python train_from_scratch.py                 # fresh run
    python train_from_scratch.py --resume ckpt.pt # resume from a checkpoint
"""
import argparse
import math
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_linear import LoRALinear

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TARGET_MODULES = ["q_proj", "v_proj"]   # attach LoRA to these attention projections
R, ALPHA = 8, 16
MAX_SEQ_LEN = 512
MICRO_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4                    # effective batch = 2 * 4 = 8
TOTAL_STEPS = 40
WARMUP_STEPS = 5
LR = 2e-4
CKPT_PATH = "checkpoint.pt"


# ---------------------------------------------------------------------------
# Step 0: attach LoRA to a real model
# ---------------------------------------------------------------------------
def attach_lora(model, target_modules, r, alpha):
    """Walk the model, replace nn.Linear layers whose name matches a target
    with a LoRALinear wrapper. Returns the list of newly-trainable params."""
    lora_params = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if any(t in child_name for t in target_modules) and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, child_name, wrapped)
                lora_params.extend(wrapped.trainable_parameters())
    return lora_params


def build_batches(dataset, tokenizer, micro_batch_size, max_len):
    """Very simple fixed-size batcher — good enough for a learning script."""
    batches = []
    for i in range(0, len(dataset) - micro_batch_size + 1, micro_batch_size):
        texts = dataset[i:i + micro_batch_size]["text"]
        enc = tokenizer(texts, return_tensors="pt", padding=True,
                         truncation=True, max_length=max_len)
        enc["labels"] = enc["input_ids"].clone()
        batches.append(enc)
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)

    # ---- Component 1 (part of it): freeze everything, attach LoRA ----
    # NOTE: attach_lora runs BEFORE model.to(device) on purpose — the new A/B
    # parameters it creates default to CPU, so we need model.to(device) to run
    # afterward to sweep them onto the GPU along with everything else.
    for p in model.parameters():
        p.requires_grad_(False)
    lora_params = attach_lora(model, TARGET_MODULES, R, ALPHA)
    model.to(device)
    n_trainable = sum(p.numel() for p in lora_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable (LoRA) params: {n_trainable:,} / {n_total:,} "
          f"({100 * n_trainable / n_total:.3f}%)")

    # ---- Component 3: activation checkpointing ----
    # This is the same torch.utils.checkpoint primitive under the hood;
    # HF's wrapper handles threading it through every decoder layer's forward
    # correctly (rotary embeddings, KV cache flags, etc.) which is fiddly to
    # hand-roll safely on a full model — so we call the built-in switch here.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # required when base weights are frozen

    # ---- Data ----
    ds = load_dataset("mlabonne/guanaco-llama2-1k", split="train[:200]")
    batches = build_batches(ds, tokenizer, MICRO_BATCH_SIZE, MAX_SEQ_LEN)
    print(f"Built {len(batches)} micro-batches "
          f"(effective batch size = {MICRO_BATCH_SIZE * GRAD_ACCUM_STEPS})")

    # ---- Component 4: optimizer + LR schedule (LoRA params only) ----
    optimizer = torch.optim.AdamW(lora_params, lr=LR)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, TOTAL_STEPS - WARMUP_STEPS)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0

    # ---- Component 5: resume from checkpoint if requested ----
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        lora_state = ckpt["lora_state"]
        for p, saved in zip(lora_params, lora_state):
            p.data.copy_(saved)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"]
        print(f"Resumed from {args.resume} at step {start_step}")

    # ---- Training loop ----
    model.train()
    step = start_step
    micro_step = 0
    optimizer.zero_grad()

    while step < TOTAL_STEPS:
        batch = batches[micro_step % len(batches)]
        batch = {k: v.to(device) for k, v in batch.items()}

        # ---- Component 2 (part a): mixed precision ----
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            out = model(**batch)
            loss = out.loss / GRAD_ACCUM_STEPS   # ---- Component 2 (part b): scale for accumulation

        loss.backward()
        micro_step += 1

        if micro_step % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            print(f"step {step:3d}/{TOTAL_STEPS} | loss {loss.item() * GRAD_ACCUM_STEPS:.4f} "
                  f"| lr {scheduler.get_last_lr()[0]:.2e}")

            # ---- Component 5: checkpoint every 10 optimizer steps ----
            if step % 10 == 0 or step == TOTAL_STEPS:
                torch.save({
                    "lora_state": [p.data.clone().cpu() for p in lora_params],
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                }, CKPT_PATH)
                print(f"  -> checkpoint saved to {CKPT_PATH}")

    print("\nDone. You can now:")
    print(f"  - inspect {CKPT_PATH} to see what a checkpoint actually contains")
    print(f"  - re-run with --resume {CKPT_PATH} to prove resume works")


if __name__ == "__main__":
    main()