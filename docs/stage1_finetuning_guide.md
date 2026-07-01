# Stage 1 — Fine-Tuning on a 24 GB RTX 5090 Mobile (Blackwell / SM120)

**Goal of this guide:** get you from a clean WSL2 environment to a working, evaluated QLoRA fine-tune that runs reliably on your 24 GB mobile 5090 — and to have you *understand* every layer, not just run a notebook.

**How it's structured (two tracks, in order):**
1. **Track A — Understand:** hand-build a minimal LoRA training loop in raw PyTorch (≈1 sitting). This is so the framework never feels like magic.
2. **Track B — Do it for real:** run a production QLoRA fine-tune in Unsloth, evaluate it, and export it to GGUF so it drops straight into your existing llama.cpp stack.

> **Epistemic labeling used throughout:** 🟢 **Official** (vendor docs/release notes), 🟡 **Strong community** (maintainer GitHub issues, reproduced reports), 🔴 **Anecdotal/contested**. Version numbers move monthly on Blackwell — treat every pin as *"verify at install."* This matches your own verify-before-you-commit habit.

---

## 0. The one-paragraph mental model

Fine-tuning teaches an existing base model a **behavior/format/domain**, not new facts at scale. **QLoRA** = load the frozen base model in 4-bit (NF4) to slash VRAM, then train only small **low-rank adapter** matrices (LoRA) on top in bf16. You're updating ~0.1–1% of the parameters. That's why an 8B model that would need ~60–120 GB to *fully* fine-tune trains comfortably in ~8–12 GB as QLoRA. The tradeoff: adapters are less expressive than full fine-tuning, but for instruction/style/domain adaptation they're usually more than enough.

---

## 1. Environment setup — this is the make-or-break part

**90% of Blackwell fine-tuning pain is environment, not training.** Get this exactly right and the rest is easy.

### 1.1 Non-negotiable rules for Blackwell (SM120)

| Rule | Why | Status |
|---|---|---|
| **Train in WSL2 (Ubuntu 24.04), not native Windows** | Building a proper sm_120 stack on Windows is unsupported; WSL2 is the supported path. | 🟢/🟡 |
| **CUDA 12.8 or 12.9 — never CUDA 13.0 / `cu130`** | `bitsandbytes 0.49.2` is built against cu12; a `cu130` torch breaks it (`libnvJitLink.so.13 not found`, `cdequantize_blockwise_fp32` symbol errors). | 🟡 (Unsloth #5154, bnb #1937) |
| **Do NOT `apt install nvidia-cuda-toolkit`** | Ubuntu's apt package is CUDA 12.0 — its `nvcc` rejects `compute_120`. Install CUDA 12.8/12.9 from NVIDIA's repo instead. | 🟡 (multiple WSL2 guides) |
| **Use `torch ≥ 2.11.0` on `cu128`/`cu129`** | `torch 2.10.0+cu128` shipped without correct sm_120 cuBLAS kernels → `CUBLAS_STATUS_EXECUTION_FAILED` on the first matmul. 2.11.0+cu129 fixes it. | 🟡 (Unsloth #5154) |
| **`bitsandbytes 0.49.2+`** | This is the version that ships prebuilt sm_120 cubins for NF4 4-bit. | 🟢 (HF install guide) |

### 1.2 Pick your install path (tiered)

| Path | Effort | Robustness | Best for |
|---|---|---|---|
| **A. Docker (`unsloth/unsloth` image)** | Lowest | **Highest** — the image is pre-built for Blackwell | Getting a guaranteed win fast; reproducibility |
| **B. Native `uv` venv** | Medium | Good | Your CLI-first preference + wanting to see the moving parts |
| **C. Unsloth Studio installer** | Low | High | If B fights you; ships Blackwell-tuned kernels via `curl -fsSL unsloth.ai/install.sh \| sh` |

I recommend **starting with Path A to prove your hardware works end-to-end**, then rebuilding with **Path B** so you own the stack. Below is the native path.

### 1.3 Native install (Path B)

**Prereqs (once):**
```bash
# In Windows PowerShell — confirm WSL2 + driver passthrough
wsl --status                 # must say Version 2
nvidia-smi                   # run INSIDE wsl afterward; must list your 5090 + CUDA 12.9
```

**Inside WSL2 Ubuntu 24.04:**
```bash
# 1) Install CUDA Toolkit 12.9 from NVIDIA's repo (NOT apt's nvidia-cuda-toolkit).
#    Follow NVIDIA's "CUDA Toolkit 12.9 — WSL-Ubuntu" instructions, then:
nvcc --version               # must show release 12.9 (or 12.8), NOT 12.0

# 2) Modern Python env manager (fits your CLI-first workflow)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv-ft && source .venv-ft/bin/activate

# 3) PyTorch on cu128/cu129 — pin >= 2.11.0 to dodge the cuBLAS bug
uv pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu129
#   (If cu129 wheels lag, use cu128; the rule is CUDA 12.x, never 13.0.)

# 4) Core training stack
uv pip install unsloth unsloth_zoo
uv pip install "bitsandbytes>=0.49.2"
uv pip install transformers trl peft accelerate datasets

# 5) (Optional, faster) xformers built for sm_120 — otherwise PyTorch SDPA is used
#    Only do this if you want the speed; SDPA works fine to start.
# pip uninstall xformers -y
# export TORCH_CUDA_ARCH_LIST="12.0"
# pip install ninja && git clone --depth=1 https://github.com/facebookresearch/xformers --recursive
# cd xformers && python setup.py install && cd ..
```

> ⚠️ **Watch for silent `cu130` upgrades.** Some packages (notably FlashAttention-4, certain vLLM builds) will *drag torch up to cu130* as a dependency and break bitsandbytes. After any install, re-run the check in §1.4. If torch moved to `+cu130`, reinstall `torch==2.11.0+cu129`.

### 1.4 Verify the stack BEFORE downloading any model

Save as `verify_stack.py` and run it. **Do not proceed to a real run until every line passes.** This is the step that separates a wasted afternoon from a smooth one.

```python
import torch
print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA not visible — check WSL2 passthrough"
cc = torch.cuda.get_device_capability(0)
print("GPU:", torch.cuda.get_device_name(0), "| compute cap:", cc)
assert cc == (12, 0), f"Expected sm_120, got {cc}"
assert "13" not in torch.version.cuda, "torch is on CUDA 13 — will break bitsandbytes!"

# bf16 matmul must run without CUBLAS error
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
y = (x @ x).float().sum()
assert torch.isfinite(y), "matmul produced non-finite — bad kernels"
print("bf16 matmul: OK")

import bitsandbytes as bnb
print("bitsandbytes:", bnb.__version__)
print("ALL BASIC CHECKS PASSED ✅")
```

### 1.5 The NF4 smoke-test (verify the contested bit yourself)

Because you'll see forum claims that "bitsandbytes gives garbage on sm_120" (🔴 — those are about **INT8 inference in vLLM**, not **NF4 QLoRA training**), verify NF4 training on *your* box with a 1-step run before committing to anything long:

```python
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",   # tiny, fast smoke target
    max_seq_length=1024, load_in_4bit=True, dtype=None,
)
model = FastLanguageModel.get_peft_model(model, r=8,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_alpha=8, use_gradient_checkpointing="unsloth", random_state=3407)

ds = load_dataset("mlabonne/guanaco-llama2-1k", split="train[:50]")
trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=ds,
    args=SFTConfig(dataset_text_field="text", max_seq_length=1024,
        per_device_train_batch_size=2, max_steps=2, logging_steps=1,
        optim="adamw_8bit", report_to="none", output_dir="smoke"))
out = trainer.train()
print("loss finite:", out.training_loss == out.training_loss)  # False if NaN
print("NF4 QLoRA TRAINING WORKS ON YOUR 5090 ✅")
```

If loss is a real finite number that moves, you're clear. (Verify dataset name `mlabonne/guanaco-llama2-1k` exists before download; it's a well-known ~1k-row test set.)

---

## 2. Track A — Build a LoRA loop from scratch (understand the mechanics)

Do this once, on a small model, in raw PyTorch. The point isn't performance — it's that LoRA stops being a black box.

**The whole idea in four lines of math.** A frozen weight `W` (shape `d_out × d_in`) gets an additive low-rank update:
```
h = W x  +  (alpha / r) * (B @ (A @ x))
#   frozen         trainable:  A is r×d_in, B is d_out×r,  r << d
```
Only `A` and `B` train. `r` sets capacity; `alpha/r` scales the update.

**Minimal `LoRALinear`:**
```python
import torch, torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():   # freeze W and bias
            p.requires_grad_(False)
        d_out, d_in = base.weight.shape
        self.A = nn.Parameter(torch.randn(r, d_in) * (1/r**0.5))
        self.B = nn.Parameter(torch.zeros(d_out, r))   # start at 0 => no-op at init
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T) @ self.B.T
```

**The five things a real training loop must do** (implement each by hand once):
1. **Mixed precision** — wrap the forward in `torch.autocast("cuda", dtype=torch.bfloat16)`.
2. **Gradient accumulation** — divide loss by `accum_steps`, only `optimizer.step()` every N micro-batches. This is how you simulate a big batch on 24 GB.
3. **Activation checkpointing** — `torch.utils.checkpoint.checkpoint(...)` to trade compute for memory.
4. **Optimizer + LR schedule** — `AdamW` on *only* the LoRA params, cosine (or linear) schedule with warmup.
5. **Checkpoint/resume** — save `{model_lora_state, optimizer, scheduler, step}` and reload it.

**Best companion resource** (🟢): Sebastian Raschka, *Build a Large Language Model (From Scratch)*, chapters 6–7 (instruction fine-tuning). Repo: `github.com/rasbt/LLMs-from-scratch`. His Appendix E implements LoRA from scratch and lines up exactly with what you'll do in Track B.

**Advance criterion for Track A:** you can point at your loss curve and explain, line by line, what accumulation, autocast, and the `alpha/r` scale are doing.

---

## 3. Track B — The real QLoRA fine-tune (Unsloth)

### 3.1 Choose the model (VRAM budget for 24 GB)

| Model size | Method | Approx. VRAM (24 GB target) | Verdict on your mobile 5090 |
|---|---|---|---|
| 0.5–1.5B | **Full** fine-tune | ~10–18 GB | ✅ Do one, to feel full FT |
| 7–8B | **QLoRA** (4-bit) | ~8–12 GB | ✅ **Your sweet spot** — headroom for longer seq/batch |
| 13–14B | **QLoRA** | ~18–23 GB | 🟡 Works with care (short seq, batch 1, grad ckpt) |
| 20B+ | QLoRA | >24 GB typical | 🔴 Cloud or the 32 GB desktop 5090 |

> **Reminder:** NVIDIA's "20B/40B on a single 5090" figures are the **32 GB desktop** card. On 24 GB, treat 8B as your default and 14B as the ceiling for comfortable work.

**Recommended starter model:** `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` or `unsloth/Qwen3-8B-bnb-4bit`. Pre-quantized `-bnb-4bit` weights download faster and load cleanly.

### 3.2 Pick the learning vehicle (dataset)

**Get a win first, then get personal:**
1. **First successful run:** a small, clean public instruction set — e.g. a 2k slice of `mlabonne/FineTome-100k` (the set Unsloth's own notebooks use). Small = you iterate in minutes.
2. **Then make it yours:** curate a **small (1k–5k) domain instruction set from your chemical-engineering / process world** (unit-operations Q&A, reaction-condition reasoning, a tool-calling/function set). This is where fine-tuning gets *judgeable* — you can actually tell if the output is right, which makes your eval honest. Building this set is itself the systematic-curation kind of task you enjoy, and it seeds a consistent throughline into Stages 2–4.

> I deliberately won't hand you a named "process-engineering instruction dataset" — I can't verify one exists at quality, and pointing you at a 404 wastes your time. Curating your own is the better learning move anyway.

### 3.3 The training script

```python
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MAX_SEQ = 2048

# 1) Load base in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
)

# 2) Attach LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,                       # 8/16/32 typical; higher = more capacity + VRAM
    lora_alpha=16,              # common heuristic: alpha == r
    lora_dropout=0.0,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    use_gradient_checkpointing="unsloth",   # big VRAM saver on 24 GB
    random_state=3407,
)

# 3) Data → chat template → "text" column
tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
ds = load_dataset("mlabonne/FineTome-100k", split="train[:2000]")
ds = standardize_sharegpt(ds)
def fmt(ex):
    return {"text": [tokenizer.apply_chat_template(c, tokenize=False,
                     add_generation_prompt=False) for c in ex["conversations"]]}
ds = ds.map(fmt, batched=True)

# 4) Train — start with max_steps to validate, then switch to epochs
trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=ds,
    args=SFTConfig(
        dataset_text_field="text", max_seq_length=MAX_SEQ,
        per_device_train_batch_size=2,      # effective batch = 2 * 4 = 8
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,                       # ← first run: keep tiny to prove it works
        # num_train_epochs=1,               # ← switch to this for the real run
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",                 # 8-bit optimizer saves VRAM
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer.train()
```

> **TRL version churn hedge:** if `SFTConfig` rejects `dataset_text_field` or `max_seq_length`, move those two args onto `SFTTrainer(...)` instead. The API for where these live has flip-flopped across TRL releases — 🟡.

### 3.4 The knobs that matter on 24 GB (and which way to turn them)

| If you hit OOM… | Turn this | Direction |
|---|---|---|
| First resort | `per_device_train_batch_size` | ↓ to 1 (raise `gradient_accumulation_steps` to keep effective batch) |
| Long inputs | `max_seq_length` | ↓ (1024 often plenty for instructions) |
| Still tight | `r` (LoRA rank) | ↓ (16 → 8) |
| Memory not compute | `use_gradient_checkpointing` | ensure `"unsloth"` |
| Optimizer memory | `optim` | keep `adamw_8bit` (not full adamw) |
| Fragmentation | env var | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

**Quality knobs** (once it runs): `learning_rate` 1e-4–2e-4 for LoRA; `r`/`alpha` up for harder domains; 1–3 epochs (watch for overfitting — see §4).

---

## 4. Evaluate (don't skip this — it's the whole point)

Fine-tuning without eval is just vibes. Do all three:

1. **Loss curves.** Training loss should fall smoothly. Hold out ~5–10% as an eval split and watch **eval loss** — if train keeps dropping while eval turns up, you're **overfitting** (fewer epochs, more data, or lower `r`).
2. **Qualitative side-by-side.** Run the **base** model and your **fine-tuned** model on the *same* 15–20 held-out prompts and read them next to each other. On a domain you understand (your chem-eng set), you can actually judge correctness — that's why the domain choice matters.
3. **A cheap quantitative check.** For a format/instruction task, script a pass/fail (does it emit valid JSON? does it follow the template? exact-match on short answers?). Even a crude accuracy number beats none.

**Overfitting tells:** verbatim regurgitation of training phrasing, degraded performance on anything slightly off-distribution, eval loss rising.

---

## 5. Save, merge, and run it in *your* llama.cpp stack

This ties Stage 1 back into the inference setup you already run.

```python
# A) Adapters only (small, portable)
model.save_pretrained("lora_model"); tokenizer.save_pretrained("lora_model")

# B) Merge to 16-bit (a standalone HF model)
model.save_pretrained_merged("model_merged_16bit", tokenizer,
                             save_method="merged_16bit")

# C) GGUF for llama.cpp — drops into your existing inference workflow
model.save_pretrained_gguf("model_gguf", tokenizer,
                           quantization_method="q4_k_m")
```

Then load the GGUF with your usual llama.cpp invocation and sanity-check generation. Being able to run your own fine-tune through the stack you already use is a genuinely satisfying end-to-end close on Stage 1.

---

## 6. When to stay local vs. reach for cloud (Stage 1 scope)

| | Condition | Action |
|---|---|---|
| 🟢 **Green** | QLoRA ≤ 8B; full FT ≤ ~1.5B; iterating on data/hyperparams | Stay on the 5090 |
| 🟡 **Yellow** | QLoRA 13–14B, or you want bigger batch/longer context and keep OOMing after §3.4 | Optimize first; rent a single H100/H200 only if it still won't fit |
| 🔴 **Red** | Full fine-tune of ≥7B, or QLoRA ≥ 20B | Cloud (1×H100 ≈ US$2.89–3.29/hr) — but you don't need this for Stage 1 learning |

For Stage 1, you should be **almost entirely green.** Cloud becomes genuinely necessary in Stage 3 (pretraining), not here.

---

## 7. Troubleshooting — exact errors you may see, and fixes

| Symptom | Cause | Fix | Src |
|---|---|---|---|
| `CUBLAS_STATUS_EXECUTION_FAILED` on first matmul | `torch 2.10.0+cu128` lacks sm_120 cuBLAS kernels | Upgrade to `torch 2.11.0+cu129`, then `rm -rf ~/.cache` compiled caches | 🟡 #5154 |
| Same error persists after torch upgrade | Stale Unsloth compiled cache | `rm -rf /tmp/unsloth_compiled_cache ./unsloth_compiled_cache` | 🟡 #5154 |
| `CUDA out of memory` **with 25+ GB free** | `torch.compile`/Inductor Triton kernels failing to launch on sm_120 | `export TORCHDYNAMO_DISABLE=1 UNSLOTH_COMPILE_DISABLE=1` | 🟡 #5154 |
| `libnvJitLink.so.13 not found`, `cdequantize_blockwise_fp32` | Something bumped torch to **cu130**, breaking bnb | Reinstall `torch==2.11.0+cu129`; keep the whole stack on CUDA 12.x | 🟡 #5154, #1937 |
| `nvcc fatal: Unsupported gpu architecture 'compute_120'` | You installed apt's CUDA 12.0 toolkit | Remove it; install CUDA 12.8/12.9 from NVIDIA | 🟡 |
| `no kernel image is available for execution` | torch wheel has no sm_120 cubins | Install a `cu128`/`cu129` wheel, `torch ≥ 2.11.0` | 🟢/🟡 |
| xformers version conflict warnings | xformers built for a different torch/CUDA | Uninstall xformers and use SDPA, or rebuild with `TORCH_CUDA_ARCH_LIST="12.0"` | 🟡 |
| "Garbage output" fears from forums | Those reports = **INT8 inference in vLLM**, not NF4 training | Ignore for QLoRA training; your §1.5 smoke-test already proved NF4 works | 🔴→resolved |

---

## 8. Success checklist — you're done with Stage 1 when…

- [ ] `verify_stack.py` and the NF4 smoke-test both pass on your 5090
- [ ] You hand-built a LoRA loop and can explain accumulation, autocast, and `alpha/r`
- [ ] A QLoRA fine-tune of an 8B model completed with a clean falling loss curve
- [ ] You ran a **held-out** eval (curves + qualitative + one quantitative metric) and saw measurable improvement over the base
- [ ] You diagnosed (or ruled out) overfitting from the eval-loss curve
- [ ] You exported to GGUF and ran your fine-tune through your own llama.cpp stack
- [ ] You did it **twice**: once on a public set (the win) and once on your own domain set

When those are checked, you'll have the exact loop that Stage 2 (distillation) reuses — distillation is essentially this SFT loop plus a teacher forward pass and a KL term. You'll be building on solid ground.

---

### Key resources (verify repo names before cloning — you know the drill)
- 🟢 `github.com/rasbt/LLMs-from-scratch` — from-scratch LoRA (Appendix E), instruction FT (ch. 6–7)
- 🟢 Unsloth docs: *Fine-tuning LLMs with Blackwell, RTX 50 series & Unsloth*
- 🟢 Unsloth *Requirements* page — VRAM-by-model table
- 🟡 Unsloth issue **#5154** — the definitive Blackwell 5090 "what broke / what fixed it" thread
- 🟡 bitsandbytes issue **#1937** — the cu130 incompatibility
- 🟢 HuggingFace TRL docs — `SFTTrainer`/`SFTConfig` reference

*All version pins are current as of mid-2026 and will drift — re-check release notes before each fresh install.*
