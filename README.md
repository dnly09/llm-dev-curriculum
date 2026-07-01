# LLM Development Curriculum — RTX 5090 Mobile (24GB, Blackwell/SM120)

End-to-end, hands-on curriculum: **Fine-tune → Distill → Pre-train from scratch → Capstone**,
built entirely on a 24GB mobile 5090, escalating to rented cloud GPUs only past defined thresholds.

Full rationale and reference material: [`docs/curriculum_plan.md`](docs/curriculum_plan.md) and
[`docs/stage1_finetuning_guide.md`](docs/stage1_finetuning_guide.md).

## Stages

| Stage | Status | Directory |
|---|---|---|
| 0. Environment setup | 🔲 not started | [`environment/`](environment/) |
| 1. Fine-tuning (LoRA/QLoRA) | 🔲 not started | [`stage1_finetuning/`](stage1_finetuning/) |
| 2. Distillation | 🔲 not started | [`stage2_distillation/`](stage2_distillation/) |
| 3. Pre-training from scratch | 🔲 not started | [`stage3_pretraining/`](stage3_pretraining/) |
| 4. Capstone (all three + scaled cloud run) | 🔲 not started | [`stage4_capstone/`](stage4_capstone/) |

Track status in [`PROGRESS.md`](PROGRESS.md).

## Ground rules (from the curriculum doc)

- Every stage: **build the core mechanic in raw PyTorch first**, then redo it in a production
  framework (Unsloth/Axolotl/torchtune → TRL → nanoGPT-lineage → TorchTitan).
- Pin **CUDA 12.8/12.9**. Never let anything drag torch to **cu130** — it breaks bitsandbytes.
- Train inside **WSL2 Ubuntu 24.04**, not native Windows.
- Cloud triggers (don't rent before you hit these):
  - 🟢 stay local: QLoRA ≤8B, full FT ≤~1.5B, offline-logit distillation, pretraining ≤~350M params/≤2-3 days
  - 🟡 optimize first, then rent: 24-40GB jobs after 4-bit+offload still don't fit
  - 🔴 cloud immediately: full FT ≥7B, pretraining ≥1B params, anything needing real multi-GPU

## Repo conventions

- Each stage folder has `from_scratch/` (raw PyTorch, no framework magic) and a framework folder.
- Every training run gets a short **report** (config, loss curve, eval results, what you'd change)
  committed alongside the code — this is what makes the eval "honest" rather than vibes.
- `environment/verify_stack.py` must pass before starting any GPU work in a fresh env.
