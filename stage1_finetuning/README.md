# Stage 1 — Fine-tuning

Full guide: `../docs/stage1_finetuning_guide.md`

- `from_scratch/` — hand-built `LoRALinear` + training loop in raw PyTorch (Track A)
- `unsloth/` — production QLoRA runs via Unsloth/TRL (Track B)
- `eval/` — held-out eval scripts + results (loss curves, qualitative comparisons, quant metrics)

Order: `from_scratch/` first, then `unsloth/`, evaluating every run in `eval/`.
Don't skip eval — see guide §4.

## Results

Four runs, three eval methodologies, in increasing rigor:

| Run | Model | Method | Final eval loss | Extra signal |
|---|---|---|---|---|
| FineTome (general chat) | Llama-3.1-8B | QLoRA | 0.727 | Qualitative: roughly a coin flip vs. base (expected — general chat has little gap to close) |
| Full FT comparison | Qwen2.5-1.5B | Full fine-tune | 0.722 | Confirmed full-FT converges as cleanly as QLoRA; ~10x smaller LR needed |
| Function calling | Llama-3.1-8B | QLoRA | 0.095 | **Scripted pass/fail: 82.5% → 90.0% exact argument match** vs. base |

See `../../assets/` for eval loss curves and the function-calling score comparison.

**Key lesson:** general-instruction fine-tuning on an already-instruction-tuned base showed little measurable edge — the base model had no real gap to close. Function calling, a narrower task with a specific format and less pretraining emphasis on precision, showed a clear, scriptable improvement. Task/dataset choice matters more than most hyperparameters for whether a fine-tune's impact is even visible.