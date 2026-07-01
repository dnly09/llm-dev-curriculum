# Environment setup notes

Full context: `../docs/stage1_finetuning_guide.md` §1.

## Non-negotiable rules

1. Train in **WSL2 Ubuntu 24.04**, not native Windows.
2. **CUDA 12.8 or 12.9 — never 13.0.** bitsandbytes 0.49.2 is built against cu12;
   cu130 breaks it (`libnvJitLink.so.13 not found`, `cdequantize_blockwise_fp32` errors).
3. Do **not** `apt install nvidia-cuda-toolkit` (that's CUDA 12.0, rejects `compute_120`).
   Install from NVIDIA's own repo.
4. Use `torch >= 2.11.0` on cu128/cu129 (2.10.0+cu128 has broken sm_120 cuBLAS kernels).
5. `bitsandbytes >= 0.49.2` (first version with prebuilt sm_120 cubins).

## Install (Path B: native `uv` venv)

```bash
# Windows PowerShell, confirm passthrough first
wsl --status                 # must say Version 2
# then inside WSL:
nvidia-smi                   # must list the 5090 + CUDA 12.9

# Inside WSL2 Ubuntu 24.04:
# 1) CUDA Toolkit 12.9 from NVIDIA's repo (not apt)
nvcc --version                # must show 12.8 or 12.9, NOT 12.0

# 2) uv + venv
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv-ft && source .venv-ft/bin/activate

# 3) PyTorch cu128/cu129, >= 2.11.0
uv pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu129

# 4) Core stack
uv pip install unsloth unsloth_zoo
uv pip install "bitsandbytes>=0.49.2"
uv pip install transformers trl peft accelerate datasets
```

⚠️ **Watch for silent cu130 upgrades** — some packages drag torch to cu130 as a dependency.
Re-run `verify_stack.py` after every install. If torch moved to `+cu130`, reinstall
`torch==2.11.0+cu129`.

## Order of operations

1. `python verify_stack.py` — must pass completely.
2. `python nf4_smoke_test.py` — confirms NF4 QLoRA training (not just inference) works.
3. Freeze the working set: `uv pip freeze > requirements.lock`.
4. Only then move to `stage1_finetuning/`.

## Troubleshooting quick reference

See `../docs/stage1_finetuning_guide.md` §7 for the full table. Most common:

| Symptom | Fix |
|---|---|
| `CUBLAS_STATUS_EXECUTION_FAILED` on first matmul | Upgrade to torch 2.11.0+cu129, clear compiled caches |
| `CUDA out of memory` with 25+GB free | `export TORCHDYNAMO_DISABLE=1 UNSLOTH_COMPILE_DISABLE=1` |
| `libnvJitLink.so.13 not found` | Something bumped you to cu130 — reinstall torch==2.11.0+cu129 |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | Remove apt's CUDA toolkit, install 12.8/12.9 from NVIDIA |
