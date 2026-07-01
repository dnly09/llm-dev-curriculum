"""
Run this BEFORE downloading any model. Every line must pass.

Usage:
    source .venv-ft/bin/activate
    python environment/verify_stack.py
"""
import torch

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not visible — check WSL2 passthrough"

cc = torch.cuda.get_device_capability(0)
print("GPU:", torch.cuda.get_device_name(0), "| compute cap:", cc)
assert cc == (12, 0), f"Expected sm_120 (Blackwell), got {cc}"
assert "13" not in torch.version.cuda, "torch is on CUDA 13 — will break bitsandbytes!"

# bf16 matmul must run without a CUBLAS error (the known torch 2.10.0+cu128 bug)
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum()
assert torch.isfinite(y), "matmul produced non-finite — bad kernels"
print("bf16 matmul: OK")

import bitsandbytes as bnb
print("bitsandbytes:", bnb.__version__)

print("ALL BASIC CHECKS PASSED ✅")
