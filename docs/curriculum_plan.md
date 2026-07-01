# End-to-End LLM Development Curriculum on an RTX 5090 Mobile (24GB Blackwell / SM120)

## TL;DR
- **Your reordering is right, and I'm tightening it:** **Fine-tune (LoRA/QLoRA) → Distill → Pre-train from scratch → Capstone.** Inside each stage, *build the core mechanic in raw PyTorch first, then re-do it in a production framework.* This matches true difficulty, respects prerequisite skills (distillation is "SFT + a teacher + a new loss"; pretraining adds tokenizer/data/architecture/stability ownership), and every non-capstone stage fits in 24GB if you size deliberately.
- **The Blackwell training stack is now mature enough to work entirely locally** (mid-2026): PyTorch ≥2.7 (use 2.9–2.11) **cu128/cu129** wheels, CUDA Toolkit **12.8/12.9 pinned**, **bitsandbytes 0.49.2** (ships sm_120 cubins — QLoRA NF4 works), Unsloth via its Blackwell Docker image, current transformers/TRL/PEFT/accelerate. **Do NOT move torch to cu130/CUDA 13.0** — it breaks bitsandbytes. flash-attention usually must be built from source or pulled from community wheels; PyTorch SDPA is a safe fallback.
- **Rent cloud only at defined thresholds:** when a job still needs >24GB after QLoRA+offload, when you want a Chinchilla-optimal pretrain above ~0.5–1B params, or when a local run would exceed ~2–3 days wall-clock. A single H100 (~$2.89–3.29/hr) handles most scaling exercises; an 8×H100 node (~$22–32/hr) reproduces a full GPT-2-grade pretrain in hours.

---

## Key Findings

### 1. Stage-ordering verdict (your proposal is sound — adopt the refined version)

Your original sequence (distill → fine-tune → pre-train → all) inverts the difficulty curve. **My refined order is correct on three independent axes — difficulty, prerequisite skills, and 24GB feasibility:**

| Stage | Why it sits here | New skills introduced | Fits 24GB? |
|---|---|---|---|
| **1. Fine-tune (LoRA/QLoRA)** | Lowest prerequisites; frameworks own the training loop; tiny datasets | Data formatting, PEFT, mixed precision, eval | ✅ 7–8B QLoRA easily |
| **2. Distill** | = an SFT loop + a teacher forward pass + a modified loss; **requires Stage 1's loop** | KL/temperature loss, teacher/student pairing, offline logit caching | ✅ with small pairs or offline logits |
| **3. Pre-train from scratch** | You own tokenizer, data pipeline, architecture, optimizer, LR schedule, **stability**, eval; longest wall-clock | BPE training, data sharding, from-zero convergence, throughput tuning | ✅ at 10M–~350M params |
| **4. Capstone (all three)** | Integration project; only the *scaled* variant needs cloud | Pipeline orchestration, end-to-end eval | Partially (base + distill local; scaled base on cloud) |

The one change I make to your phrasing: **distillation is genuinely "mid" difficulty only because it leans on Stage 1.** If you tried it first (your original order) it would feel harder than fine-tuning, because you'd be debugging a KL loss *and* a training loop *and* two-model VRAM pressure simultaneously. Doing fine-tuning first removes two of those three unknowns. This is the strongest argument for the reorder.

---

## Details

### 2. Blackwell / SM120 tooling (CRITICAL — verify at install time; this moves fast)

**The governing fact:** Blackwell sm_120 (compute capability 12.0) requires **CUDA 12.8+** *and* a PyTorch wheel actually compiled with sm_120 cubins. Per NVIDIA's CUDA 12.8 release ("CUDA Toolkit Now Available for NVIDIA Blackwell") and the Blackwell Compatibility Guide, native sm_120 cubins require CUDA Toolkit 12.8+. **PyTorch 2.7.0 was the first stable release with native Blackwell support** — the official PyTorch 2.7 blog states it added "support for the NVIDIA Blackwell GPU architecture and pre-built wheels for CUDA 12.8 across Linux x86 and arm64… PyTorch 2.7 includes Triton 3.3, which adds support for the Blackwell architecture with torch.compile compatibility." The current recommended build is 2.11.0 on the cuda12.8 tag (2.7.0 minimum).

#### Recommended, currently-working stack (epistemic status labeled)

| Component | Version / action | Source quality |
|---|---|---|
| **NVIDIA driver (Windows host)** | ≥576.x exposing CUDA 12.9 to WSL2 | Community (WSL2 setup guides) |
| **PyTorch** | 2.9–2.11, **cu128 or cu129** wheel | **Official** (PyTorch 2.7 blog; Salad docs) |
| **CUDA Toolkit** | **12.8 or 12.9** (do NOT install Ubuntu's `nvidia-cuda-toolkit` — it's CUDA 12.0 and rejects compute_120) | Official + community |
| **bitsandbytes** | **0.49.2** — prebuilt Linux/Windows **cu128/cu129** wheels include sm_120 cubins; NF4 4-bit QLoRA works | **Official** HF install guide |
| **Unsloth** | Use the **`unsloth/unsloth` Docker image** (Blackwell-ready); or pip-install and build xformers from source with `TORCH_CUDA_ARCH_LIST="12.0"` | **Official** (Unsloth docs + NVIDIA blog) |
| **transformers / TRL / PEFT / accelerate** | Latest stable | Official |
| **Triton** | 3.3+ (bundled with torch 2.7+) | Official |
| **flash-attention** | **No reliable official sm_120 prebuilt wheels** — build from source (`USE_FLASH_ATTENTION` disabled on first build to avoid OOM, then add back) or use community wheels (e.g., KBlueLeaf/5090wheels, marcorez8 Blackwell wheels). PyTorch **SDPA is the safe fallback.** | Community |
| **xformers** | Optional; build from source with `TORCH_CUDA_ARCH_LIST="12.0"` | Official Unsloth guide |

#### The single most important pitfall (verified by subagent + community)
**Do NOT advance torch to cu130 / CUDA 13.0 while using bitsandbytes.** Although bitsandbytes does publish a CUDA 13.0 wheel, the shipped 0.49.2 PyPI build linked against the cu12 toolkit breaks under a cu130 runtime with `libnvJitLink.so.13 not found` and `cdequantize_blockwise_fp32` symbol errors (documented in Unsloth issue #5154 and bnb issue #1937). **Pin the whole stack to CUDA 12.8/12.9.** This directly affects your QLoRA stage.

#### Other documented Blackwell gotchas
- **cuBLAS execution failures** (`CUBLAS_STATUS_EXECUTION_FAILED` on the first matmul) on torch 2.10.0+cu128 — fixed by upgrading to a build with proper sm_120 cuBLAS kernels (Unsloth #5154, torch 2.11.0+cu129). *Community.*
- **torch.compile / Inductor Triton kernels** can fail to launch on sm_120 (manifests as spurious OOM) — interim workaround `TORCHDYNAMO_DISABLE=1` / `UNSLOTH_COMPILE_DISABLE=1`; clear stale `unsloth_compiled_cache`. *Community.*
- **"Garbage output" on sm_120 is NOT a bitsandbytes-NF4 problem.** The subagent confirmed those reports trace to *other* libraries' FP4/FP8/MXFP4 block-scaled-MMA paths (NVIDIA/cutlass #3096, sglang #21132, llama.cpp #19662, vLLM INT8). bitsandbytes NF4 blockwise dequant + standard matmul does not use those instructions and works correctly. *Distinguish carefully: this myth will mislead you otherwise.*
- **MoE 4-bit limitation:** bitsandbytes cannot yet quantize 3D fused expert tensors (some MoE architectures), blocking QLoRA on those — dense models are fine. *Community.*

#### FP8 / FP4 training — what's actually usable
- **FP8 training: usable now.** Use **NVIDIA Transformer Engine** (official FP8 + MXFP8/NVFP4 support on Blackwell) or **torchao** (PyTorch-native, vendor-neutral FP8, wired into `accelerate` and TorchTitan). Community report (karpathy/nanochat discussion #382): **20–30% throughput gains from FP8 vs nanochat on RTX 5090**, with NVFP4 adding up to ~20% more on some configs (but slower on others). Unsloth ships FP8 GRPO/RL for consumer GPUs.
- **FP4 training: experimental.** NVFP4/MXFP4 *inference* works in vLLM/TRT-LLM/SGLang, but FP4 **MoE GEMM kernels produce garbage on desktop sm_120** (CUTLASS #3096 — desktop Blackwell reports plain `compute_120`, not the `compute_120a` variant whose block-scaled MMA instructions FP4 paths need). Treat FP4 training as a research curiosity for now.

#### WSL2 specifics
- **Train in WSL2, not native Windows** — building PyTorch with proper sm_120 support on Windows is unsupported; WSL2 is the supported path (community consensus across multiple setup guides).
- Increase WSL memory limit in `.wslconfig` before compiling xformers/flash-attn; use `--no-build-isolation`; keep `MAX_JOBS=4` (≈30GB RAM) to avoid OOM-killed CUDA compiles.
- Verify `nvidia-smi` inside WSL2 shows your driver + CUDA 12.9 before anything else.

---

### 3. Stage 1 — Fine-tuning

**Learning objectives:** the full data→train→eval loop; LoRA/QLoRA math; mixed precision; gradient accumulation/checkpointing; checkpoint/resume; overfitting diagnosis.

**From-scratch first (≈1 week):**
- Implement a `LoRALinear` module by hand: frozen `W`, trainable low-rank `B@A`, scaling `α/r`, then `y = xW + (α/r)·x B A`.
- Hand-roll the loop: bf16 `autocast`, manual gradient accumulation, `torch.utils.checkpoint` activation checkpointing, AdamW, cosine LR with warmup, and a manual save/resume of optimizer+scheduler+step.
- Best companion: **Sebastian Raschka, *Build a Large Language Model (From Scratch)*** (Manning, ISBN 978-1633437166; repo `github.com/rasbt/LLMs-from-scratch`) — chapters 6–7 cover classification and instruction fine-tuning from raw PyTorch. Verified repo name.

**Then production frameworks:**

| Framework | Ease | VRAM efficiency | Blackwell maturity | Config | Lock-in | Coverage | Multi-GPU (later cloud) |
|---|---|---|---|---|---|---|---|
| **Unsloth** | High (notebook/CLI) | **Best** (custom Triton kernels; ~30–50% less VRAM, ~2× faster on single GPU) | **Strong** — official Blackwell Docker + NVIDIA blog | Python | Medium | LoRA/QLoRA, full FT, GRPO, FP8 RL | **Single-GPU only** (multi-GPU gated to Pro) |
| **Axolotl** | High (YAML) | Good (FlashAttn, grad ckpt defaults) | Good (HF-stack based) | **YAML** (reproducible) | Low | LoRA/QLoRA, full FT, QAT, seq-parallel, RLHF | **Yes** (best for distributed) |
| **torchtune** | Medium (recipes) | Good (24GB-tested recipes) | Good (PyTorch-native) | Python recipes | **Lowest** (pure PyTorch) | LoRA/QLoRA, full FT, QAT | Yes (FSDP) |
| **TRL** | Medium (building block) | Good | Good | Python | Low | SFT/DPO/GRPO/**GKD**/GOLD | Yes (accelerate) |

**My recommendation for you (CLI-first, anti-lock-in):** Learn on **Unsloth** for the fast local feedback loop, but keep **Axolotl** (YAML, reproducible, multi-GPU) as your "production + cloud-scaling" tool, and use **torchtune** when you want to read/modify the actual loop (closest to your from-scratch work, least lock-in). TRL is the connective tissue you'll reuse in Stage 2.

**VRAM budget on 24GB (training, not inference):**
- Full fine-tune needs ~weights + gradients + optimizer states + activations ≈ 12–16× param-bytes; a 7B full FT needs ~60–120GB → **cloud only.**
- **QLoRA 7–8B: ~8–12GB** → fits with comfortable headroom on 24GB, room for longer sequences/larger batch.
- **QLoRA 13–14B: fits with care** (gradient checkpointing, short seq, small batch).
- **Full fine-tune locally: realistic only ~0.5–1.5B** params (e.g., Qwen3-0.6B/1.7B, Llama-3.2-1B) — do at least one so you experience full FT, not just PEFT.

**Recommended learning vehicle/dataset:** Lean into your chemical-engineering background — a **small (1k–10k example) instruction dataset in a scientific/process-engineering niche** (e.g., unit-operation Q&A, reaction-condition reasoning, or a function-calling/tool-use set). It's small enough to iterate in minutes on 24GB, gives you a domain you can *judge quality on* (critical for honest eval), and seeds a coherent throughline across all four stages.

---

### 4. Stage 2 — Distillation

**Conceptual grounding (objectives):**
- **By signal:** response/**logit-based** (KL between temperature-softened teacher/student distributions, classic Hinton), **feature/hidden-state-based** (MSE on intermediate states), **attention-based**.
- **By policy:** **off-policy** (student learns from static teacher outputs — exposure bias) vs **on-policy** (student generates, teacher scores its own trajectories — fixes train/inference mismatch).
- **By granularity:** sequence-level vs token-level.
- **State of the art (2024–2026):** **MiniLLM** (ICLR 2024, reverse-KL), **DistiLLM** (ICML 2024, skew KL + adaptive off-policy), **GKD** (Generalized KD, on-policy, JSD with λ/β knobs), and 2026 work like **GOLD** (cross-tokenizer on-policy via Universal Logit Distillation). Note: TRL has a documented bug where GKD/MiniLLM fail under *different* teacher/student tokenizers (issue #4562) — use GOLD/ULD or a shared tokenizer.

**From-scratch first (≈1 week):** teacher + student forward passes; compute `KL(softmax(z_t/T) || softmax(z_s/T)) · T²`; combine as `L = α·CE(student, labels) + (1−α)·KL`; sweep temperature `T`. This is the cleanest way to *feel* what distillation transfers.

**Then frameworks:**

| Tool | Approach | Notes |
|---|---|---|
| **TRL `GKDTrainer`** | On-policy GKD (forward/reverse KL, JSD; λ mixes on/off-policy, β interpolates) | Subclasses `SFTTrainer`; the easiest "real" KD entry. Import from `trl.experimental.gkd`. |
| **TRL GOLD** | On-policy + cross-tokenizer (ULD) | `trl.experimental` — newest; works across model families. |
| **Arcee DistillKit** | **Logit-based + hidden-state**, online & offline, top-k sparse logits | `github.com/arcee-ai/DistillKit`. Powers Arcee's released models. Best for offline-logit workflows. Verified repo. |

**VRAM on 24GB — the key tradeoff:**
- **Online (both models resident):** pick a small pair, e.g., teacher ~3–7B (4-bit) → student ~0.5–1.5B. DistillKit's own note: "memory requirements for distillation are higher compared to standard SFT."
- **Offline (recommended on 24GB):** precompute the teacher's top-k logits to disk once, then train the student alone — removes the teacher from VRAM entirely. DistillKit supports this with logit compression (polynomial approx + quantization + bit-packing).
- **Recommended pairing for learning:** distill **Qwen2/3-1.5B-Instruct (teacher) → 0.5B (student)** (the canonical DistillKit setup), or distill *your Stage-1 fine-tuned 7–8B teacher → a 1B student* to create a continuous narrative.

---

### 5. Stage 3 — Pre-training from scratch (hardest)

**Objectives:** tokenizer training; data sharding/streaming; architecture (RoPE, RMSNorm, QK-norm); optimizer (AdamW, and modern Muon); LR warmup/cooldown; *stability*; throughput; from-zero eval.

**From-scratch path (in order of increasing realism):**
1. **Karpathy *Neural Networks: Zero to Hero*** (`karpathy.ai/zero-to-hero.html`, repo `github.com/karpathy/nn-zero-to-hero`) — micrograd → makemore → "Let's build GPT" → the tokenizer (minBPE). This is the foundational from-scratch curriculum.
2. **`github.com/karpathy/nanoGPT`** (now superseded but still the cleanest ~300-line GPT + ~300-line training loop) and **`karpathy/build-nanogpt`** (the GPT-2 reproduction video/repo). Reference **`karpathy/llm.c`** for the C/CUDA implementation (GPT-2 124M to 3.28 val loss in ~45 min on 8×H100).
3. **`github.com/KellerJordan/modded-nanogpt`** (the "NanoGPT speedrun") — teaches *modern* training tricks: rotary embeddings, QK-norm, ReLU², the **Muon optimizer**, FP8 head, FlexAttention, attention-window warmup. Current record trains GPT-2-grade in **~2 min 20 s on 8×H100 / <400M tokens.** Best single resource for "what's changed since 2019."
4. **`github.com/karpathy/nanochat`** — the **end-to-end** successor (tokenizer→pretrain→midtrain→SFT→RL→inference→web UI, ~8,000 LOC). This is effectively your Stage-4 capstone template; the "$100 speedrun" trains a GPT-2-grade chat model in ~4 hrs on 8×H100.

**Realistic scale on a single 24GB Blackwell GPU (hours–days):**
- **Tiny (minutes–hours):** 1M–50M params on **TinyStories** (Eldan & Li, arXiv 2305.07759) — a synthetic ~3–4-year-old-vocabulary corpus where even sub-10M-param models produce coherent English; a great fast-iteration target ("less than a day on a single GPU" in the original work).
- **GPT-2 small class (1–3 days):** ~124M–350M params on the **FineWeb-Edu 10B-token sample** (the dataset modded-nanogpt/llm.c use) or OpenWebText. The original llm.c GPT-2 124M took ~4 days on a single 8×A100 *node*; a single 24GB card targets the smaller end of this with a reduced token budget.
- **Token budget:** Chinchilla-optimal is ~20 tokens/param, but for *learning* on small models, deliberate over-training (TinyStories-style) is fine and instructive. Decide explicitly: a compute-optimal 350M run wants ~7B tokens — feasible locally only over days, which is exactly your **cloud trigger**.

**Tokenizer:** train your own **BPE** at least once (nanochat uses a Rust BPE with a 65,536 vocab on FineWeb-Edu; minBPE teaches the algorithm). Then practice *reusing* GPT-2's tokenizer to feel the tradeoff (vocab fit vs convenience; modded-nanogpt notes GPT-2's vocab has ~240 dead tokens on FineWeb).

**Production pretraining frameworks (for cloud scaling):**

| Framework | Owner | Strength | Learn it when |
|---|---|---|---|
| **litgpt** | Lightning AI | 20+ LLM impls, pretrain+finetune recipes, readable | Bridging from nanoGPT to "real" recipes |
| **TorchTitan** | PyTorch | PyTorch-native 4D parallelism (FSDP2+TP+PP), FP8, async checkpoint | **Your primary cloud-scaling target** (lowest lock-in, native) |
| **nanotron** | Hugging Face | Minimalistic 3D parallelism | Studying parallelism cleanly |
| **GPT-NeoX** | EleutherAI | Megatron+DeepSpeed combined | Established multi-node recipes |
| **Megatron-LM / Megatron-Core** | NVIDIA | Maximum-scale TP/PP/DP, newest arch support | Only at true cluster scale |

**Recommendation:** For your anti-lock-in preference, **TorchTitan** is the right "scale-up" framework to learn (PyTorch-native, composable parallelism, FP8 via torchao), with litgpt as the gentler on-ramp.

---

### 6. Stage 4 — Capstone (all three together)

**Design — "Small Domain LLM, end to end":**
1. **Pretrain** a ~50–150M base from scratch on a domain-flavored mix (e.g., FineWeb-Edu + a scientific/process-engineering corpus) using your nanoGPT/nanochat code locally. Train your own BPE tokenizer on the mix.
2. **Distill** capability into it (or into a smaller sibling) from a strong open teacher (e.g., a Qwen3/Llama-3 instruct model) using DistillKit offline-logit distillation — fits 24GB because the teacher is precomputed.
3. **Fine-tune** (QLoRA via Unsloth, or full FT since the base is small) on your Stage-1 instruction set to make it follow domain instructions.
4. **Evaluate** end-to-end with a held-out domain eval you can judge, plus a generic check (e.g., a small slice of standard benchmarks).
5. **Scaled variant (cloud):** repeat step 1 at ~2× params / compute-optimal tokens on a rented 1×H100 or 8×H100 node using TorchTitan or nanochat's `speedrun.sh`, then redo 2–4. This is exactly the "scale a model by 2× params" exercise you wanted, and nanochat is purpose-built for it.

---

### 7. Cloud scaling — explicit decision thresholds

**🟢 GREEN — stay local on the 24GB 5090:**
- QLoRA/LoRA of models ≤8B (≤~14B with care).
- Full fine-tune ≤~1.5B.
- Distillation with offline teacher logits, or small online teacher/student pairs.
- Pretraining ≤~350M params with a reduced token budget and ≤~2–3 days wall-clock.

**🟡 YELLOW — optimize locally first, rent if it doesn't fit:**
- Job needs 24–40GB: try 4-bit + gradient checkpointing + CPU/NVMe optimizer offload (DeepSpeed ZeRO-Offload) + shorter sequences. If still OOM or painfully slow → rent a single **H100 80GB / H200 141GB**.
- Pretraining a *compute-optimal* 350M–1B model (token budget pushes wall-clock past ~3 days) → rent.

**🔴 RED — go to cloud immediately:**
- Full fine-tune of ≥7B, or any ≥13B full FT.
- Pretraining ≥1B params or any compute-optimal run needing tens of B tokens.
- Anything requiring genuine multi-GPU (learning FSDP/TP/PP at realistic scale).
- Your "2× params" capstone scaling exercise.

**Provider guidance (rates are snapshots — verify live):**

| Provider | Best for | Indicative price (2026) | Notes |
|---|---|---|---|
| **RunPod** | On-demand single + multi-node, Blackwell-ready templates | H100 PCIe ~**$2.89/hr**, SXM5 ~**$3.29/hr**; 8×H100 SXM ≈ **$21.5/hr**; B200 on-demand ~$5.89/hr, RTX 4090 ~$0.34/hr | Per-second billing, zero egress, startup credits (up to 1,000 free H100 hrs). Largest community marketplace. |
| **Lambda** | Cleanest fixed pricing, pre-built stack | H100 PCIe **$3.29/hr**; 8×H100 SXM **$3.99/GPU-hr** (≈$31.9/hr) with 22 TiB local SSD included | On-demand only; ships Lambda Stack (PyTorch 2.7.0, CUDA 12.8, NCCL 2.26.2). This is the box Karpathy's nanochat speedrun used (~$24/hr historically). |
| **Vast.ai** | Cheapest commodity GPUs (marketplace) | RTX 4090 / A100 lowest, live-priced | Reliability varies by host; verify at deploy. |
| **Spheron** | Cheap H100/B200 spot | H100 SXM5 ~$2.50/hr on-demand (~$1.03 spot); B200 spot ~$2.12/hr | Good for checkpoint-friendly batch pretraining. |

**Distributed-training learning path (the actual point of going to cloud):**
1. **FSDP / FSDP2** (PyTorch-native sharding of params/grads/optimizer) — start here; it's what TorchTitan and torchtune use.
2. **DeepSpeed ZeRO** (stages 1→3, + Offload) — the classic memory-sharding mental model; ZeRO-Offload also helps you in YELLOW locally.
3. **Tensor + Pipeline parallelism (TP/PP)** via TorchTitan/Megatron — only once data-parallel sharding is second nature.
Practice progression: 1×H100 (port your code) → 2×H100 (FSDP) → 8×H100 (FSDP+TP, run the nanochat/modded-nanogpt speedrun).

---

## Recommendations (staged action plan)

**Phase 0 — Environment (1–2 days):** In WSL2/Ubuntu, create a uv/conda env: PyTorch cu128 (2.9–2.11), bitsandbytes 0.49.2, transformers/TRL/PEFT/accelerate, Unsloth Docker image. **Pin CUDA 12.8/12.9; never cu130.** Validate: `torch.cuda.get_device_name()`, a tiny bf16 matmul, and a 1-step QLoRA smoke test. Keep a `requirements.lock`.

**Phase 1 — Fine-tune (2–3 weeks):** (a) Hand-code a LoRA loop on a ≤1.5B model (Raschka ch. 6–7). (b) Reproduce it in Unsloth QLoRA on a 7–8B model with your scientific instruction set. (c) Do one *full* fine-tune of a ~1B model to feel the difference. (d) Re-run the same job in Axolotl (YAML) and torchtune to compare. **Benchmark to advance:** you can explain every line of your from-scratch loop and your eval shows measurable gain over the base.

**Phase 2 — Distill (2 weeks):** (a) Hand-code KL+CE distillation on a tiny pair. (b) Run TRL `GKDTrainer` (on-policy). (c) Run DistillKit **offline-logit** distillation (teacher precomputed to disk) — distill your Phase-1 7–8B teacher → 1B student. **Benchmark:** the 1B student beats its own from-scratch SFT baseline on your domain eval.

**Phase 3 — Pre-train (3–4 weeks):** (a) Work through Zero-to-Hero + "Let's build GPT." (b) Train a ~10–30M model on TinyStories to convergence (hours). (c) Train a ~124M GPT-2-class model on FineWeb-Edu 10B sample locally (budget-limited), studying modded-nanogpt's tricks (Muon, RoPE, QK-norm). Train your own BPE once. **Benchmark:** coherent TinyStories generation; a falling val-loss curve on FineWeb that you can interpret.

**Phase 4 — Capstone (2–3 weeks):** Run the pretrain→distill→fine-tune pipeline locally on a small base; then **rent an 8×H100 node** and reproduce the scaled (2× params, compute-optimal) version with nanochat/TorchTitan, learning FSDP→TP along the way. **Benchmark:** an end-to-end report (à la nanochat `report.md`) for both local and scaled runs.

**Thresholds that change the plan:** if any local job OOMs after 4-bit+offload, or any single run projects >3 days → escalate to YELLOW/RED cloud immediately rather than fighting it locally.

---

## Caveats
- **Time-sensitivity:** Blackwell training tooling changes monthly. Treat every version number here as "verify at install." The most stable invariants are: *CUDA 12.8/12.9 pinned, sm_120 needs an sm_120-compiled wheel, and don't pair cu130 with bitsandbytes.* Re-check PyTorch, bitsandbytes, and Unsloth release notes before each phase.
- **Source quality:** PyTorch 2.7 sm_120 support, NVIDIA's CUDA 12.8 requirement, and bitsandbytes' sm_120 cubin targeting are **official**. The cuBLAS/torch.compile failures, the cu130-breaks-bnb finding, and FP8 speedup figures are **strong community reports** (GitHub issues, maintainer threads) — reliable but not vendor-guaranteed. flash-attn community wheels are **unofficial**; prefer SDPA if you hit symbol errors.
- **Hardware note:** your **mobile** 5090 has **24GB**; many cited 5090 benchmarks (e.g., NVIDIA/Unsloth's "40B on a single Blackwell GPU," 31B QLoRA at ~22GB) use the **32GB desktop** part. Budget conservatively against 24GB — subtract ~2–4GB of those headlines.
- **Repo names verified:** `rasbt/LLMs-from-scratch`, `karpathy/nn-zero-to-hero`, `karpathy/nanoGPT`, `karpathy/build-nanogpt`, `karpathy/llm.c`, `karpathy/nanochat`, `KellerJordan/modded-nanogpt`, `arcee-ai/DistillKit`, `Lightning-AI/litgpt`, `NVIDIA/Megatron-LM`, `huggingface/trl` (GKD/GOLD), `NVIDIA/TransformerEngine`. Confirm the exact clone URL before any large download, since forks with similar names exist.
- **Distillation tokenizer trap:** TRL GKD/MiniLLM mis-handle *different* teacher/student tokenizers (issue #4562) — use a shared tokenizer or the GOLD/ULD path.