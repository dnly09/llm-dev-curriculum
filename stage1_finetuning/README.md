# Stage 1 — Fine-tuning

Full guide: `../docs/stage1_finetuning_guide.md`

- `from_scratch/` — hand-built `LoRALinear` + training loop in raw PyTorch (Track A)
- `unsloth/` — production QLoRA runs via Unsloth/TRL (Track B)
- `eval/` — held-out eval scripts + results (loss curves, qualitative comparisons, quant metrics)

Order: `from_scratch/` first, then `unsloth/`, evaluating every run in `eval/`.
Don't skip eval — see guide §4.
