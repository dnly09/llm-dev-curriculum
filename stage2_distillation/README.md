# Stage 2 — Distillation

**Status:** In progress (Track A complete, Track B Tier 1 next)

Full walkthrough: [`docs/stage2_distillation_guide_5090mobile.md`](../docs/stage2_distillation_guide_5090mobile.md)
Live checklist: [`PROGRESS.md`](../PROGRESS.md)

## Goal

Take a large, capable **teacher** and compress its behavior into a small, cheap
**student** that runs fast in the existing llama.cpp stack — then prove, with
the same scriptable eval built in Stage 1, that distillation beats plain SFT
at the same student size.

Distillation is treated as "SFT plus a teacher signal": it reuses Stage 1's
QLoRA/SFT pipeline (`train_qlora.py`, `train_full_ft.py`,
`score_function_calling.py`) rather than building a new one from zero.

## Environment

Distillation runs in its **own venv** (`.venv-distill`), separate from Stage
1's `.venv-ft`, because vLLM/DistillKit pull their own pinned
`torch`/`transformers` and shouldn't be allowed to disturb the already-locked
Stage 1 environment.

| Component | Version |
|---|---|
| torch | 2.11.0+cu128 |
| bitsandbytes | 0.49.2 |
| transformers | 5.12.1 |
| trl | 1.7.0 |

Lockfile: [`../requirements.distill.lock`](../requirements.distill.lock)

**Phase 0 gate** (`phase0_distill_gate.py`) confirmed before any trainer code
was written:
- `GKDConfig`/`GKDTrainer` import from `trl.experimental.gkd` on this TRL version
- `GKDConfig` uses `max_length` (not `max_seq_length` — same rename Stage 1 hit
  on `SFTConfig`)
- `lmbda`/`beta`/`seq_kd` all present on `GKDConfig`
- `TRL_EXPERIMENTAL_SILENCE=1` is set in `.venv-distill/bin/activate` to quiet
  the recurring experimental-API warning

## Model pairing

| Role | Model | Why |
|---|---|---|
| Teacher | `Qwen/Qwen2.5-7B-Instruct` (4-bit NF4) | Strong at tool-calling; shares tokenizer lineage with small Qwen students |
| Student | `Qwen/Qwen2.5-0.5B-Instruct` | Matches teacher's chat-template tokenizer; small enough to fully fine-tune |
| Task | `hiyouga/glaive-function-calling-v2-sharegpt` | Same throughline as Stage 1; scriptable eval already exists |

**Gotcha resolved during Track A:** Qwen2.5 checkpoints pad the
`lm_head`/embedding matrix to different widths **by model size**, independent
of base-vs-instruct — the 0.5B pads to `vocab_size=151936`, the 7B pads to
`152064`, even though the real tokenizer vocab (`len(tokenizer)`) is smaller
than both (~151,665) and identical across the family. Any logit-KD code
truncates both teacher and student logits to `len(tokenizer)` before
computing KL — see `kd_loss_scratch.py` for the pattern.

## Track A — from scratch (done)

`kd_loss_scratch.py` hand-implements the Hinton KD loss
(`alpha·KL(student‖teacher) + (1−alpha)·CE`), does one real teacher+student
forward pass on the Qwen pair, sweeps `T ∈ {1,2,4} × alpha ∈ {0,0.5,0.9}`
read-only to observe the loss components, then runs **one real training
step** at `T=2.0, alpha=0.5` with an actual `.backward()`/`.step()`.

Key lesson from the sweep: the `T²` rescale in the KD term isn't there to
shrink the loss as temperature rises — it's there to keep gradient magnitudes
comparable across temperatures. The reported `kd_term` can rise or plateau
with `T` depending on how the rescale trades off against the raw KL shrinkage
from softening the distributions.

## Track B — the four tiers (in progress)

Done in order; each tier adds one new capability over the last.

| Tier | Method | Status |
|---|---|---|
| **1** | Sequence-level KD — teacher generates completions offline, student does plain SFT on them (reuses Stage 1's `train_qlora.py`/`train_full_ft.py`) | ⏳ next |
| **2** | Offline logit KD via DistillKit — teacher's top-k logits captured to disk, student trained with teacher never co-resident | ⏳ |
| **3** | Online on-policy KD via TRL `GKDTrainer` — student generates, teacher scores, fixes exposure bias | ⏳ |
| **4** (stretch) | Cross-tokenizer distillation via GOLD/ULD | ⏳ optional |

## Evaluation

Same `score_function_calling.py` harness from Stage 1, run head-to-head on a
held-out split across: base student (floor) → student + plain SFT (the bar
to beat) → student + distillation (the payoff) → teacher (ceiling). Success
looks like **distilled student > SFT student**, closing a meaningful fraction
of the gap to the teacher.

## Export

Same `export_gguf.py` path as Stage 1 — served through the GPU llama.cpp
build (not Unsloth's CPU-only drop at `~/.unsloth/llama.cpp`).

## Next: Stage 4 capstone thread

A strong function-calling student here makes function calling the natural
capstone domain (pretrain a small base → distill toward function calling →
fine-tune) — worth keeping in mind when Stage 3's pretraining dataset choices
come up.