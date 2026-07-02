# Stage 2 — Knowledge Distillation on a 24 GB RTX 5090 Mobile (Blackwell / SM120)

**Goal of this guide:** take a large, capable **teacher** and compress its behavior into a small, cheap **student** that runs fast in your llama.cpp stack — and prove, with your existing scriptable eval, that distillation beats plain SFT at the same student size.

**Prerequisite:** Stage 1 complete (you have a working QLoRA/SFT pipeline, `score_function_calling.py`, and the `glaive-function-calling-v2-sharegpt` data). Stage 2 *reuses* that pipeline; distillation is largely "SFT plus a teacher signal."

> **Epistemic labels:** 🟢 **Official** · 🟡 **Strong community / maintainer** · 🔴 **Contested/anecdotal**. All version pins are mid-2026 and drift monthly on Blackwell — verify at install, per your own habit.

---

## 0. Carry-forward from your Stage 1 session (read first)

These are your own logged lessons, mapped to where they bite again in Stage 2. Handling them here pre-empts the exact debugging rounds you flagged.

| Stage 1 finding | Stage 2 consequence | Action baked into this guide |
|---|---|---|
| `SFTConfig` uses **`max_length`**, not `max_seq_length` (cost you 2 debugging rounds) | `GKDConfig` **extends `SFTConfig`**, so it inherits `max_length`. Same rename applies. | All configs below use `max_length`. §1.3 has you dump the signature once. |
| Unsloth silently downgraded **torch** (2.11→2.10) on install | Installing `trl`/DistillKit/vLLM can re-trigger this, breaking sm_120 cuBLAS | §1.2 re-verifies `torch.__version__` after every install |
| **transformers** version ambiguous (lock said 5.12.1, banner said 5.5.0) — never resolved | `GKDTrainer`/`DistillationTrainer`/`GOLD` live in `trl.experimental` and are **version-sensitive** | §1.3 makes pinning + import-testing `trl` and `transformers` a **hard Phase 0 gate** |
| Task choice mattered more than hyperparams; **function calling** gave real signal (82.5%→90%) | Distillation's effect is only *visible* on a checkable task | Guide uses function calling as the throughline and reuses `score_function_calling.py` |
| Full-FT LR **2e-5** (~10× smaller than LoRA) converged cleanly | Small students (0.5–1.5B) are often **full-trained** in distillation | §4 uses 2e-5 as the full-student default |
| GGUF lands in `model_gguf_gguf/`; Unsloth ships a **CPU-only** llama.cpp | Same export path for the distilled student | §6 reuses `export_gguf.py` and re-checks the folder |
| **Stage 4 capstone domain is open** (chem-eng dropped) | — | §7 notes: a strong function-calling student here makes it the natural capstone domain |

---

## 1. Environment — additions on top of Stage 1

Your Stage 1 env (WSL2 Ubuntu 24.04, `uv` venv `.venv-ft`, torch 2.11.0+cu128, bnb 0.49.2, unsloth 2026.6.9, CUDA 12.8) is the base. Distillation adds a teacher, and optionally vLLM (for fast teacher generation / offline logit capture) and a distillation toolkit.

**Strong recommendation:** do Stage 2 in a **fresh venv** (`.venv-distill`), not `.venv-ft`. vLLM and DistillKit pull their own pinned `torch`/`transformers`, and you do **not** want that stomping the Stage 1 env you've already locked. Copy `requirements.lock` aside first.

### 1.1 What to add

```bash
uv venv --python 3.12 .venv-distill && source .venv-distill/bin/activate

# Re-pin the Blackwell-safe base FIRST (same rules as Stage 1)
uv pip install --pre torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install "bitsandbytes>=0.49.2" transformers accelerate datasets peft

# TRL for GKD / DistillationTrainer / GOLD
uv pip install "trl>=0.12"

# (Tier 2) DistillKit for offline logit distillation with compression
# git clone https://github.com/arcee-ai/distillkit.git && cd distillkit && uv pip install -e . && cd ..

# (Tiers 1-3) vLLM for fast teacher generation / logit capture / teacher-server
# uv pip install vllm     # ⚠️ vLLM often bumps torch — re-verify after (see §1.2)
```

### 1.2 The recurring trap (your gotcha #1) — automate the check

vLLM and DistillKit are the most likely packages to drag `torch` to a bad version. After **every** install line above, run:

```bash
python - <<'PY'
import torch
v = torch.__version__
print("torch:", v)
assert torch.cuda.is_available(), "CUDA not visible"
assert torch.cuda.get_device_capability(0) == (12, 0), "not sm_120"
assert "cu13" not in v and "+cu13" not in v, "torch on CUDA 13 — will break bitsandbytes!"
print("torch OK ✅")
PY
```

If it moved: `uv pip install --pre torch==2.11.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128` and clear stale caches (`rm -rf /tmp/unsloth_compiled_cache ./unsloth_compiled_cache`).

### 1.3 Phase 0 gate — resolve your transformers/TRL open thread *now*

This is the gate. Do not write a trainer script until all four print cleanly:

```python
# phase0_distill_gate.py
import transformers, trl, torch
print("transformers:", transformers.__version__)   # decide deliberately; don't let it "ride"
print("trl:", trl.__version__)

# Which import path does YOUR trl expose? (moved to experimental on recent main)
try:
    from trl.experimental.gkd import GKDConfig, GKDTrainer
    print("GKD import: trl.experimental.gkd ✅")
except ImportError:
    from trl import GKDConfig, GKDTrainer            # older stable path
    print("GKD import: trl (stable) ✅")

# Confirm the SFTConfig arg name (your Stage 1 max_length lesson) — GKDConfig inherits it
import inspect
sig = inspect.signature(GKDConfig.__init__)
print("has max_length:", "max_length" in sig.parameters,
      "| has max_seq_length:", "max_seq_length" in sig.parameters)
print("GKDConfig lmbda/beta/seq_kd present:",
      all(k in sig.parameters for k in ("lmbda","beta","seq_kd")))
```

> **Pin decision:** if `transformers` resolves to something old that predates a trainer you need, upgrade it *deliberately* and re-run this gate. Write the resolved versions into a `requirements.distill.lock` and commit it, exactly as you did for Stage 1.

---

## 2. The mental model (one page)

**Distillation = train a small student to imitate a big teacher.** Three families, easiest → hardest, all usable on 24 GB:

| Family | What the student learns from | On/Off-policy | Tokenizer match needed? | VRAM cost |
|---|---|---|---|---|
| **Sequence-level KD** | Teacher's **generated text** (then plain SFT on it) | Off-policy | ❌ No (it's just text) | Lowest — teacher runs *offline* |
| **Token/logit KD** | Teacher's **full probability distribution** per token (KL) | Off-policy | ✅ **Yes** (shared vocab) | Med — teacher logits (can precompute) |
| **On-policy KD (GKD)** | Teacher feedback on the **student's own** generations | On-policy | ✅ Yes (or GOLD/ULD) | Highest — both models live + generation |

**Why on-policy is the SOTA direction:** off-policy KD trains the student on prefixes it never produces itself, so at inference the student drifts into states it was never taught to recover from (**exposure bias / distribution mismatch**). On-policy KD (GKD) supervises the student *on its own rollouts*, which is the DAgger-style fix. 🟢 (TRL GKD docs; Agarwal et al.)

**The divergence knob (`beta` in GKD):** `beta=0` ≈ forward KL (student covers all teacher modes, can be "blurry"); `beta=1` ≈ reverse KL (student commits to the teacher's dominant modes — often better for generation, per MiniLLM); in-between = generalized JSD. 🟢/🟡

**The `lmbda` knob:** `0.0` = pure off-policy (teacher probs on fixed data); `1.0` = pure on-policy (student generations); `seq_kd=True, lmbda=0` = sequence-level KD. 🟢

---

## 3. The design decision — teacher, student, task

### 3.1 The tokenizer rule decides everything
For **logit** distillation (Tiers 2–3), teacher and student **must share a tokenizer**, or the logits don't align token-for-token. The clean way to guarantee this is to stay **within one model family**.

### 3.2 Recommended pairing (fits 24 GB, same tokenizer)

| Role | Model | Why |
|---|---|---|
| **Teacher** | `Qwen/Qwen2.5-7B-Instruct` (start) → `Qwen/Qwen2.5-14B-Instruct` (stretch) | Strong at tool-calling; shares tokenizer with small Qwen students |
| **Student** | `Qwen/Qwen2.5-0.5B` or `Qwen2.5-1.5B` | You already used these in Stage 1's from-scratch + full-FT runs |
| **Task** | `hiyouga/glaive-function-calling-v2-sharegpt` | Your throughline; scriptable eval already built |

> **Why not your Llama-3.1-8B fine-tune as teacher?** Different tokenizer from Qwen students → logit KD won't align. You *can* use it for **sequence-level KD** (Tier 1, text-only) or via **GOLD/ULD** cross-tokenizer (Tier 4). If you'd rather keep Llama end-to-end, pick a small Llama student (e.g. `Llama-3.2-1B`) and a Llama teacher to keep vocab shared.

### 3.3 VRAM budget on 24 GB

| Scenario | Fits 24 GB? | Notes |
|---|---|---|
| Teacher 7B (4-bit) **generation only**, offline | 🟢 Easily | Then student trains alone |
| Precompute teacher **logits** (7–14B 4-bit) offline → train student | 🟢 | Teacher and student never co-resident |
| **Online** GKD: teacher 7B (4-bit) + student 0.5–1.5B + generation | 🟡 Tight but doable | Short `max_length`, batch 1–2, grad-ckpt |
| Online GKD with 14B teacher | 🔴 | Use offline logits, or teacher-server (§5) |
| Teacher ≥32B | 🔴 | Cloud precompute or external vLLM server |

---

## 4. Track A — Distillation loss from scratch (understand it once)

Before any framework, implement the classic **Hinton KD loss** so the `temperature`/`alpha` knobs aren't magic. This mirrors your Stage 1 `train_from_scratch.py` — same loop, plus a frozen teacher forward pass and a KL term.

```python
import torch, torch.nn.functional as F

def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    student_logits, teacher_logits: [B, S, V]  (already shifted for next-token)
    labels: [B, S] with -100 on prompt/pad positions
    T: temperature (softens distributions). alpha: weight on soft (KD) vs hard (CE).
    """
    # --- soft targets: KL(student || teacher) on softened logits ---
    s_logp = F.log_softmax(student_logits / T, dim=-1)
    t_prob = F.softmax(teacher_logits / T, dim=-1)
    kd = F.kl_div(s_logp, t_prob, reduction="none").sum(-1)      # [B, S]
    mask = (labels != -100)
    kd = (kd * mask).sum() / mask.sum() * (T * T)                # T^2 keeps grads scaled

    # --- hard targets: standard next-token CE on ground truth ---
    ce = F.cross_entropy(student_logits.flatten(0, 1),
                         labels.flatten(), ignore_index=-100)

    return alpha * kd + (1.0 - alpha) * ce
```

**The five things to get right** (all inherited from your Stage 1 loop, plus #4–5):
1. **Shift** logits/labels for next-token prediction (`[..., :-1, :]` vs `[..., 1:]`).
2. **Mask** prompt tokens with `-100` so KD only applies to completion tokens.
3. **Freeze the teacher** (`teacher.eval()`, `torch.no_grad()` on its forward) — huge VRAM saver.
4. **Temperature** `T`: 1.0–4.0; higher reveals more of the teacher's "dark knowledge" (its relative confidence across wrong answers). Start `T=2.0`.
5. **alpha**: 0.5 is a fine start; push toward the KD term (0.7–0.9) once it's stable.

**Advance criterion (Track A):** you can run one teacher+student step on `Qwen2.5-0.5B` and explain what `T` and `alpha` did to the loss. Then move to frameworks.

**Companion reading:** the original Hinton et al. "Distilling the Knowledge in a Neural Network"; MiniLLM (Gu et al. 2023) for the reverse-KL argument; the GKD paper (Agarwal et al., "On-Policy Distillation of Language Models") for the on-policy fix.

---

## 5. Track B — the four framework tiers (easiest → SOTA)

Do them in order. Tier 1 is your guaranteed win and reuses 100% of Stage 1; each later tier adds one new capability.

### Tier 1 — Sequence-level KD (start here; tokenizer-agnostic, lowest VRAM)
**Idea:** teacher generates high-quality completions → student does plain SFT on them. This is literally your Stage 1 pipeline with a teacher-authored dataset. (It's also, mechanically, what "DeepSeek-R1-Distill" style models are.)

1. **Generate** with the teacher *offline* (fits easily; teacher never co-resident with training):
   ```bash
   # via vLLM for speed; or reuse your own llama.cpp/Ollama homelab endpoint
   # feed each function-calling prompt, save teacher completions to a JSONL
   ```
   Or, if you want a *stronger* teacher than fits locally, generate on a rented GPU (§6) or via any OpenAI-compatible endpoint your homelab exposes.
2. **SFT the student** on those completions using your Stage 1 `train_qlora.py` (or full-FT `train_full_ft.py` with **LR 2e-5** for a 0.5–1.5B student).
3. **Score** with `score_function_calling.py`.

**Why start here:** zero new failure modes, tokenizer-agnostic (so your Llama-3.1-8B *can* be the teacher), and it establishes the baseline that Tiers 2–3 must beat.

### Tier 2 — Offline logit KD with DistillKit (memory-safe real distillation)
**Idea:** capture the teacher's **full (top-k) logit distribution** once, to disk, then train the student with no teacher in VRAM. DistillKit's whole reason for existing is making this storage-feasible. 🟢

- Repo (verified): `https://github.com/arcee-ai/distillkit` → `uv pip install -e .`
- **Capture** teacher logits (teacher alone; 7–14B 4-bit fits 24 GB):
  ```bash
  python -m distillkit.sample_logits_vllm \
    --model Qwen/Qwen2.5-7B-Instruct \
    --dataset <your-glaive-fc-split> \
    --output ./teacher_logits/ \
    --compression-config ./compression_config.yaml
  ```
  <cite index="29-1">DistillKit compresses stored logits with polynomial approximation + error-diffusion quantization + bit-level packing, because naively storing top-k logit pairs over billions of tokens is prohibitive.</cite>
- **Distill** the student against the captured logits. DistillKit supports **composable losses** — <cite index="29-1">KL, JSD, TVD, ranking losses, and hidden-state alignment (`hs_cosine`), in sparse (top-k) or dense mode</cite>:
  ```yaml
  loss_functions:
    - { function: cross_entropy, weight: 0.25 }
    - { function: kl, weight: 0.5, temperature: 2.0 }
    - { function: hs_cosine, weight: 0.25 }   # hidden-state matching (needs same depth/width family)
  ```
- **Same tokenizer required** for the logit/KL terms (that's why we chose Qwen→Qwen). For cross-tokenizer, DistillKit pairs with `mergekit-tokensurgeon`, but that's Tier 4 territory.

**When to prefer Tier 2 over Tier 3:** a 14B teacher, or you want reproducible runs / to iterate on student hyperparams without re-running the teacher each time.

### Tier 3 — Online on-policy KD with TRL `GKDTrainer` (the SOTA method)
**Idea:** student generates, teacher scores those generations, student learns from the feedback — fixing exposure bias. Both models live in memory (or teacher on a server, below).

```python
# import path per your Phase 0 gate (§1.3): experimental on recent main
from trl.experimental.gkd import GKDConfig, GKDTrainer   # or: from trl import ...
from transformers import AutoModelForCausalLM, AutoTokenizer

student_id = "Qwen/Qwen2.5-0.5B"
teacher_id = "Qwen/Qwen2.5-7B-Instruct"
tok = AutoTokenizer.from_pretrained(student_id)          # shared-family tokenizer

student = AutoModelForCausalLM.from_pretrained(student_id, torch_dtype="bfloat16")
teacher = AutoModelForCausalLM.from_pretrained(
    teacher_id, torch_dtype="bfloat16",
    load_in_4bit=True,                                   # 4-bit teacher to fit 24 GB
)

cfg = GKDConfig(
    output_dir="gkd_student",
    max_length=1024,                 # ← NOT max_seq_length (your Stage 1 lesson)
    per_device_train_batch_size=1,   # generation is memory-heavy; keep small
    gradient_accumulation_steps=8,
    learning_rate=2e-5,              # small full-trained student
    lmbda=0.5,                       # 0=off-policy, 1=fully on-policy student rollouts
    beta=0.5,                        # 0≈forward-KL, 1≈reverse-KL, between=JSD
    temperature=2.0,
    max_new_tokens=256,
    report_to="none",
)
trainer = GKDTrainer(model=student, teacher_model=teacher,
                     args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
```

**Tuning order (per the GKD authors):** on-policy (higher `lmbda`) tends to win, but the **best `beta` is task-dependent** — sweep `beta ∈ {0, 0.5, 1}` on your function-calling eval. 🟢

**Teacher-server escape hatch (uses your homelab):** if teacher+student+generation won't co-fit, TRL's newer `DistillationTrainer` <cite index="27-1">moves the teacher to an external vLLM server so it doesn't need to fit on the same GPU as the student, and adds a generation buffer that can speed training up to ~40×.</cite> You could run the teacher via vLLM on your MSI/RTX-3080 box (small teacher ≤~7B 4-bit given 10 GB) or a rented GPU, and train the student on the 5090. 🟢/🟡 (new API — verify the config against current docs.)

### Tier 4 — Cross-tokenizer distillation (stretch: mismatched teacher/student)
Only if you want a teacher whose tokenizer differs from the student (e.g. Llama teacher → Qwen student). TRL's **GOLD** trainer handles this. <cite index="28-1">GOLD adds Universal Logit Distillation (ULD) with a hybrid mode that compares exact vocabulary matches directly and falls back to sorted-probability ULD for unmatched tokens, inheriting GKD's on/off-policy scheduling; it lives in `trl.experimental` and its API may change.</cite>

⚠️ **Known cross-tokenizer bug to know about:** a BOS-token misalignment (e.g. Llama-3 prepends `<|begin_of_text|>`, Qwen/Phi don't) corrupted the merge/loss signal; <cite index="22-1">it was reported as TRL issue #4393 and fixed with a byte-offset walker in TRL #5885.</cite> Make sure your TRL is new enough to include that fix before trusting cross-tokenizer runs.

---

## 6. Evaluate — the one comparison that matters

Distillation is only worth it if the **distilled student beats a same-size student trained by plain SFT**. Run all four through `score_function_calling.py` on the **same held-out split**:

| Model | What it tells you |
|---|---|
| Base student (no training) | Floor |
| Student + **plain SFT** (Stage 1 recipe) | The bar distillation must clear |
| Student + **distillation** (Tier 2 or 3) | The payoff |
| Teacher | Ceiling (the gap you're closing) |

Report the same metric family you built in Stage 1 (exact-argument-match %, plus the 3-level correctness scoring). A successful Stage 2 looks like: **distilled student > SFT student**, moving a meaningful fraction of the way from the SFT bar toward the teacher ceiling. If distilled ≈ SFT, your task may be too easy for the teacher's extra signal to matter (the Stage 1 "FineTome coin-flip" lesson, re-applied) — pick harder held-out cases or a bigger teacher gap.

Reuse your Stage 1 loss-curve + qualitative harness too (`compare_base_vs_finetuned.py`, re-pointed at student vs teacher).

---

## 7. Export & the capstone thread

- **Export** the distilled student to GGUF with your Stage 1 `export_gguf.py`, and **re-check the `model_gguf_gguf/` folder quirk** (your gotcha #4) in case the current Unsloth changed it. Serve through **your** GPU llama.cpp build, not the CPU-only one Unsloth drops at `~/.unsloth/llama.cpp` (gotcha #5).
- **Capstone (Stage 4) domain:** a strong function-calling student here makes **function calling the natural capstone throughline** (pretrain a small base → distill toward function calling → fine-tune). That resolves your open thread #6 — worth deciding now so Stage 3's dataset choices point the same direction.

---

## 8. Cloud thresholds for Stage 2

| | Condition | Action |
|---|---|---|
| 🟢 **Green** | Teacher ≤ 7–8B; sequence-level or offline-logit KD; online GKD with 4-bit 7B teacher | Stay on the 5090 |
| 🟡 **Yellow** | Teacher 14B online GKD, or you want faster teacher generation | Offline-logit route locally first; or teacher on homelab/rented vLLM server |
| 🔴 **Red** | Teacher ≥ 32B, or you want a 70B-class teacher's signal | Rent a GPU to **precompute logits** (`sample_logits_vllm`) or host the teacher server; student still trains on the 5090 |

For a first pass, Tiers 1–2 keep you **green**. You only need cloud when you deliberately reach for a teacher too big to run 4-bit locally (≈1×H100 for a 70B teacher logit-capture pass).

---

## 9. Troubleshooting — Stage-2-specific

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: cannot import name 'GKDTrainer' from 'trl'` | It moved to `trl.experimental.gkd` | Use the try/except import from §1.3 |
| `GKDConfig got unexpected keyword 'max_seq_length'` | Same rename you hit in Stage 1 | Use `max_length` |
| Logit/KL loss is `NaN` or nonsense | Teacher/student **tokenizer mismatch** (vocab misaligned) | Use same-family pair, or switch to Tier 1 (seq-KD) / Tier 4 (GOLD) |
| `NaN` logits specifically with **Gemma** teacher/student | Gemma soft-capping | Set `attn_implementation="flash_attention_2"` (🟢 GKD docs) |
| OOM once generation starts (online GKD) | Student rollouts + teacher co-resident | ↓ `per_device_train_batch_size` to 1, ↓ `max_new_tokens`, 4-bit teacher, or go offline (Tier 2) |
| Cross-tokenizer run corrupts loss across whole sequence | BOS-misalignment bug | Ensure TRL includes the #5885 byte-offset fix |
| `torch` silently on cu13 after installing vLLM/DistillKit | Dependency bump (your gotcha #1) | Re-pin torch 2.11.0+cu128; re-run §1.2 |
| DistillKit distill config "doesn't match" captured logits | Compression config must match between capture and train | Reuse the *same* `compression_config.yaml` for both steps (🟢 DistillKit docs) |

---

## 10. Success checklist — advance to Stage 3 when…

- [ ] Phase 0 gate passes: `trl`/`transformers` versions **pinned and committed**, GKD import path confirmed, `max_length` confirmed
- [ ] You hand-implemented the KD loss and can explain `temperature` and `alpha`
- [ ] **Tier 1** sequence-level KD student trained and scored (the baseline)
- [ ] **Tier 2 or 3** distilled student trained (offline logits *or* online GKD)
- [ ] Head-to-head eval done: **distilled student vs plain-SFT student vs teacher**, same `score_function_calling.py`, distilled beats SFT
- [ ] Distilled student exported to GGUF and served through your **GPU** llama.cpp build
- [ ] Capstone domain decision logged (function calling recommended)

When those are checked, you'll have a small model that punches above its size — and the exact teacher→student machinery Stage 4 reuses. Stage 3 (pretraining from scratch) is next, and it's the one where cloud finally becomes genuinely useful rather than optional.

---

### Key resources (verify repo names before cloning — your standing rule)
- 🟢 TRL GKD docs — `huggingface.co/docs/trl` → *GKD Trainer* (`lmbda`/`beta`/`seq_kd`, teacher_model arg)
- 🟢 TRL `DistillationTrainer` + `GOLD` docs (both `trl.experimental`; teacher-server, ULD cross-tokenizer)
- 🟢 `github.com/arcee-ai/distillkit` — offline logit distillation w/ compression; function-calling case study
- 🟢 `github.com/pytorch/torchtune` — `recipes/knowledge_distillation_single_device.py` (LoRA KD alternative; `tune ls`)
- 🟡 `github.com/chrisliu298/awesome-on-policy-distillation` — curated 2025–2026 OPD papers/tools
- 🟢 Papers: Hinton et al. (KD), Gu et al. 2023 (MiniLLM / reverse-KL), Agarwal et al. (GKD / on-policy)

*Version pins current as of mid-2026; TRL's distillation trainers are in active `experimental` churn — re-check the config signatures each fresh install (your Phase 0 gate does this automatically).*
