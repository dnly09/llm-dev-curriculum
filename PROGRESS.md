# Progress

## Phase 0 — Environment

- [x] WSL2 Ubuntu 24.04 confirmed, nvidia-smi shows 5090 + driver
- [x] CUDA Toolkit 12.8/12.9 installed from NVIDIA repo (not apt)
- [x] uv venv created, torch cu128/cu129 ≥2.11.0 installed
- [x] bitsandbytes ≥0.49.2, transformers/TRL/PEFT/accelerate installed
- [x] environment/verify_stack.py passes
- [x] NF4 QLoRA 1-step smoke test passes
- [x] requirements.lock committed

## Stage 1 — Fine-tuning

- [x] Track A: hand-built LoRA loop (raw PyTorch), can explain accumulation/autocast/alpha-r
- [x] Track B: QLoRA fine-tune of 7-8B model, clean falling loss curve
- [x] One full fine-tune of a ~1B model (feel the VRAM difference vs QLoRA)
- [x] Held-out eval: loss curves + qualitative side-by-side + one quantitative metric
- [x] Overfitting diagnosed/ruled out
- [x] Exported to GGUF, ran through llama.cpp stack
- [x] Repeated on own domain instruction set (chem-eng / process engineering)

## Stage 2 — Distillation

- [x] Environment: `.venv-distill` created, torch 2.11.0+cu128 / bitsandbytes 0.49.2 pinned, `trl>=0.12` (resolved to 1.7.0) installed, torch re-verified after each install
- [x] Phase 0 gate passed: transformers 5.12.1, trl 1.7.0, GKD import confirmed at `trl.experimental.gkd`, `GKDConfig` uses `max_length` (not `max_seq_length`), `TRL_EXPERIMENTAL_SILENCE=1` set in venv activate script
- [x] `requirements.distill.lock` committed
- [x] Hand-coded KL+CE distillation on the real teacher/student pair (Qwen2.5-7B-Instruct → Qwen2.5-0.5B-Instruct), T/alpha sweep + one real training step
- [x] Tier 1 — sequence-level KD (teacher generates completions, student SFTs on them)
- [ ] Tier 2 — DistillKit offline-logit distillation: 7B teacher → 0.5-1.5B student
- [ ] Tier 3 — TRL `GKDTrainer` (on-policy) run
- [x] Head-to-head eval: distilled student vs plain-SFT student vs teacher, on held-out 240-example split (121 function_call examples scored)
- [ ] Distilled student exported to GGUF, served through GPU llama.cpp build
- [ ] (Stretch) Tier 4 — cross-tokenizer distillation via GOLD/ULD

**Tier 1 result (exact_args on 121 held-out function_call examples):** base_student (untrained) 86.8% | baseline_sft 90.1% | tier1_distilled 85.1% | teacher 90.9%. On its face, distillation looks like it underperformed plain SFT — but a miss-by-miss diagnostic revealed a more nuanced picture:
- 10 of baseline_sft's 12 misses and 10 of tier1_distilled's 18 misses are on the **same examples with identical predicted values** — both models independently produce the same "wrong" answer (e.g. both guess `calories=[100,50,20]` for an unstated meal, both say `'car'` instead of `'driving'`). These are shared 0.5B-scale limitations or inherently ambiguous ground truth (arbitrary "current date" fields baked into the dataset at collection time), not a distillation-specific weakness.
- The ~6 misses unique to tier1_distilled are mostly a **narrow stylistic pattern**: the teacher's generation style omits empty-placeholder optional keys (`title: ''`, default `language: 'English'`) that the dataset's own ground truth includes by convention; baseline_sft, trained directly on that ground truth, naturally matches it. This is a strict-match scoring artifact, not a capability gap.
- One unique miss (dropped "comedy" from a movie search) is a genuine content-loss error — real, but n=1, not a strong trend.

**Lesson:** exact-match scoring against one data source's labeling conventions can penalize a distilled model for learning an equally valid but stylistically different convention from the teacher. Before concluding a distillation method underperformed, diff the actual misses, not just the aggregate score. For Tier 2/3: consider either explicitly prompting the teacher to match the target dataset's placeholder-key convention during generation, or building a normalized/fuzzy scorer (case-insensitive, singular/plural-insensitive, empty-key-tolerant) for a fairer comparison.

**Known gotcha logged:** Qwen2.5 checkpoints pad `lm_head`/embedding matrices to different widths depending on **model size**, independent of base-vs-instruct — 0.5B pads to `vocab_size=151936`, 7B pads to `152064`, even though the real tokenizer vocab (`len(tokenizer)`) is smaller than both (~151665) and identical across the family. Any logit-KD code must truncate both teacher and student logits to `len(tokenizer)` before computing KL, or it hits a shape mismatch (or worse, silently compares padded noise columns if sizes happened to match by luck).

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