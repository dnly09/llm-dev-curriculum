# Progress

## Phase 0 — Environment
- [ ] WSL2 Ubuntu 24.04 confirmed, `nvidia-smi` shows 5090 + driver
- [ ] CUDA Toolkit 12.8/12.9 installed from NVIDIA repo (not apt)
- [ ] `uv` venv created, torch cu128/cu129 ≥2.11.0 installed
- [ ] bitsandbytes ≥0.49.2, transformers/TRL/PEFT/accelerate installed
- [ ] `environment/verify_stack.py` passes
- [ ] NF4 QLoRA 1-step smoke test passes
- [ ] `requirements.lock` committed

## Stage 1 — Fine-tuning
- [ ] Track A: hand-built LoRA loop (raw PyTorch), can explain accumulation/autocast/alpha-r
- [ ] Track B: QLoRA fine-tune of 7-8B model, clean falling loss curve
- [ ] One full fine-tune of a ~1B model (feel the VRAM difference vs QLoRA)
- [ ] Held-out eval: loss curves + qualitative side-by-side + one quantitative metric
- [ ] Overfitting diagnosed/ruled out
- [ ] Exported to GGUF, ran through llama.cpp stack
- [ ] Repeated on own domain instruction set (chem-eng / process engineering)

## Stage 2 — Distillation
- [ ] Hand-coded KL+CE distillation on a tiny teacher/student pair
- [ ] TRL `GKDTrainer` (on-policy) run
- [ ] DistillKit offline-logit distillation: Stage-1 7-8B teacher → 1B student
- [ ] Student beats its own from-scratch SFT baseline on domain eval

## Stage 3 — Pre-training
- [ ] Zero-to-Hero / "Let's build GPT" worked through
- [ ] Own BPE tokenizer trained
- [ ] ~10-30M model trained to convergence on TinyStories
- [ ] ~124M GPT-2-class model trained on FineWeb-Edu 10B sample
- [ ] Falling val-loss curve, interpretable

## Stage 4 — Capstone
- [ ] Local pipeline: pretrain → distill → fine-tune on small domain base
- [ ] End-to-end eval report (local run)
- [ ] Scaled run on rented 8xH100 node (2x params, compute-optimal tokens)
- [ ] End-to-end eval report (scaled run)

## Cloud spend log
| Date | Provider | GPU(s) | Hours | Cost | Purpose |
|---|---|---|---|---|---|
| | | | | | |
