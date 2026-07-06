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
- [x] Phase 0 gate passed: transformers 5.12.1 (drifted to 5.13.0 during Tier 2, gate re-passed), trl 1.7.0, GKD import confirmed at `trl.experimental.gkd`, `GKDConfig` uses `max_length` (not `max_seq_length`), `TRL_EXPERIMENTAL_SILENCE=1` set in venv activate script
- [x] `requirements.distill.lock` committed (regenerated once after a DistillKit install accidentally downgraded trl in `.venv-distill` — recovered by rebuilding the venv clean)
- [x] Hand-coded KL+CE distillation on the real teacher/student pair (Qwen2.5-7B-Instruct → Qwen2.5-0.5B-Instruct), T/alpha sweep + one real training step
- [x] Tier 1 — sequence-level KD (teacher generates completions, student SFTs on them)
- [x] Tier 2 — DistillKit offline-logit distillation: 7B teacher → 0.5B student, both filtered (2157 ex) and unfiltered (2760 ex) splits captured and trained, for direct comparison
- [ ] Tier 3 — TRL `GKDTrainer` (on-policy) run
- [x] Head-to-head eval: all six checkpoints (base_student, baseline_sft, tier1_distilled, tier2_unfiltered, tier2_filtered, teacher) scored on the same held-out 240-example split (121 function_call examples for exact_args, full 240 for call-vs-clarify F1)
- [x] Winning checkpoint (tier2_unfiltered) exported to GGUF (q4_k_m, 373MB), verified running through GPU llama.cpp build with correct JSON-only function-call output
- [ ] (Stretch) Tier 4 — cross-tokenizer distillation via GOLD/ULD

**Tier 1 result (exact_args on 121 held-out function_call examples):** base_student (untrained) 86.8% | baseline_sft 90.1% | tier1_distilled 85.1% | teacher 90.9%. On its face, distillation looks like it underperformed plain SFT — but a miss-by-miss diagnostic revealed a more nuanced picture:
- 10 of baseline_sft's 12 misses and 10 of tier1_distilled's 18 misses are on the **same examples with identical predicted values** — both models independently produce the same "wrong" answer (e.g. both guess `calories=[100,50,20]` for an unstated meal, both say `'car'` instead of `'driving'`). These are shared 0.5B-scale limitations or inherently ambiguous ground truth (arbitrary "current date" fields baked into the dataset at collection time), not a distillation-specific weakness.
- The ~6 misses unique to tier1_distilled are mostly a **narrow stylistic pattern**: the teacher's generation style omits empty-placeholder optional keys (`title: ''`, default `language: 'English'`) that the dataset's own ground truth includes by convention; baseline_sft, trained directly on that ground truth, naturally matches it. This is a strict-match scoring artifact, not a capability gap.
- One unique miss (dropped "comedy" from a movie search) is a genuine content-loss error — real, but n=1, not a strong trend.

**Lesson:** exact-match scoring against one data source's labeling conventions can penalize a distilled model for learning an equally valid but stylistically different convention from the teacher. Before concluding a distillation method underperformed, diff the actual misses, not just the aggregate score.

**Follow-up: "call vs clarify" binary classification metric (`score_call_vs_clarify.py`).** Reused the same four checkpoints, no retraining — scored a cleaner, unambiguous sub-task (does the model correctly decide to call a function at all, vs. ask a clarifying question first) on the full 240-example held-out set via precision/recall/F1. Ground truth here has no formatting ambiguity, so this sidesteps every artifact from the exact_args metric above:

| Model | Precision | Recall | F1 | Accuracy | FP (of 119 negatives) |
|---|---|---|---|---|---|
| base_student (untrained) | 50.6% | 100% | 67.2% | 50.8% | 118 |
| baseline_sft | 97.6% | 100% | 98.8% | 98.8% | 3 |
| tier1_distilled | 89.6% | 100% | 94.5% | 94.2% | 14 |
| teacher (Qwen2.5-7B-Instruct) | 67.2% | 100% | 80.4% | 75.4% | 59 |

**Tier 1 result: tier1_distilled beat its own teacher decisively** (94.5% F1 vs 80.4%) but did not beat baseline_sft. **Why:** call-vs-clarify has a clean, directly-trainable ground-truth label already in the dataset. Sequence-level KD is mechanically "SFT on a different (teacher-generated) label set" — it has no channel to transmit anything beyond a single hard text target, so it cannot, by construction, beat training on the original clean label. Only logit-level distillation (Tier 2) transmits the teacher's full probability distribution — genuine extra signal beyond a hard label. This was the reasoning for moving to Tier 2.

**Known gotcha logged:** Qwen2.5 checkpoints pad `lm_head`/embedding matrices to different widths depending on **model size**, independent of base-vs-instruct — 0.5B pads to `vocab_size=151936`, 7B pads to `152064`, even though the real tokenizer vocab (`len(tokenizer)`) is smaller than both (~151665) and identical across the family. Any logit-KD code must truncate both teacher and student logits to `len(tokenizer)` before computing KL, or it hits a shape mismatch. DistillKit's own capture script (`sample_logits_vllm.py`) does this automatically via an `auto_vocab_size` flag; our own hand-rolled capture script replicated the same truncation manually.

---

### Tier 2 — offline logit KD (complete)

**Environment note:** DistillKit's setup pulls its own pinned `trl`/`accelerate`/`huggingface-hub` versions that conflict with the Phase 0-gated versions in `.venv-distill` (downgraded trl 1.7.0 → 0.25.1). Isolated into a separate venv, `.venv-distill-tier2`, following the same pattern as the Stage 1/Stage 2 split. Optional `[capture]` extras (for vLLM-based logit capture) additionally pulled a **CUDA-13-linked vLLM build** (vLLM's PyPI default switched to cu13 as of v0.20.0) that's incompatible with the project's CUDA-12.8-pinned stack. Rather than force a cu128 vLLM variant, capture was done with a **plain `transformers` forward pass** instead (`capture_teacher_logits.py`) — DistillKit's compression/writer machinery (`LogprobCompressor`, `StreamingParquetWriter`) has no vLLM dependency, only its optional fast-generation capture script does.

**Data design decision:** Tier 2 conditions both teacher and student on the **original ground-truth conversation** (not the teacher's Tier 1 generated text) — this is what makes it logit-level KD rather than "SFT on teacher text again." Tier 1's hallucination filter was re-applied to select which *original* examples to include (not to filter generated text, since Tier 2 doesn't train on generated text) — captured both the full 2760-example split (unfiltered) and the 2157-example filtered split, to directly test whether soft-target training is more robust to the teacher's known hallucination bias than Tier 1's hard-label filtering needed to be.

**Training config:** DistillKit offline (`teacher.kind: dataset`), `cross_entropy` (weight 0.5) + `kl` (weight 0.5, T=1.0) composite loss, `k=exact_k=64` (lossless top-64, no polynomial/residual compression — dataset small enough that storage wasn't a constraint), 3 epochs, LR 2e-5, batch 8, `remove_unused_columns: false` (required — HF Trainer's default silently strips the custom `compressed_logprobs`/`bytepacked_indices` columns otherwise).

**Results (call_vs_clarify F1, the metric that actually discriminates per Tier 1's findings):**

| Model | Precision | Recall | F1 | Accuracy | FP (of 119) |
|---|---|---|---|---|---|
| baseline_sft | 97.6% | 100% | 98.8% | 98.8% | 3 |
| **tier2_unfiltered** | 97.6% | 99.2% | **98.4%** | 98.3% | 3 |
| tier1_distilled | 89.6% | 100% | 94.5% | 94.2% | 14 |
| tier2_filtered | 82.9% | 100% | 90.6% | 89.6% | 25 |
| teacher | 67.2% | 100% | 80.4% | 75.4% | 59 |

**Headline result: tier2_unfiltered essentially closes the gap to baseline_sft** (1 example apart out of 240, tied on false positives) — the mechanistic payoff predicted at the end of Tier 1: logit-level KD carries real distributional signal beyond a hard label, and here it delivered close-to-parity with clean-label SFT while training on soft targets throughout.

**tier2_filtered underperforming (90.6% F1, 25 FP vs. tier2_unfiltered's 3) has a mechanistic explanation, not just noise:** Tier 1's filter was designed for a paradigm where the teacher's *generated text* was the label — removing hallucinated completions removed bad labels. Tier 2 always trains against ground truth, so filtering doesn't touch the label at all; what it actually does is disproportionately **remove clarify-required examples** (since those are the ones the teacher was likely to hallucinate on), skewing the training distribution toward over-calling. The filtered student ends up mirroring a miniature version of the teacher's own bias. **The Tier 1 filter should not have been reused verbatim for Tier 2 — unfiltered data is the right choice for logit-level KD on this task**, resolving the open question from the Tier 2 handoff.

**exact_args stayed flat across all student variants (~86.8%, same as untrained base_student), below baseline_sft's 90.1%.** Diagnosed via `diagnose_tier2_misses.py` (extends the Tier 1 diagnostic with an automatic overlap check against baseline_sft's misses): of tier2_unfiltered's 16 misses, 10 were shared with baseline_sft (same wrong prediction, both models), and all 6 unique misses fell into the exact three artifact classes already documented in Tier 1 — singular/plural noun mismatches, empty-placeholder key omission, and arbitrary "current year" ground-truth values. **No new failure mode; exact_args is confirmed as the wrong metric to judge distillation quality on for this task**, across both tiers.

**Export:** `tier2_unfiltered` exported via llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` (q4_k_m, 373MB) — not Unsloth's `save_pretrained_gguf()`, since this checkpoint is a plain `transformers`-saved full model with no LoRA adapter to merge. Verified running through the project's own GPU-enabled llama.cpp build (402 t/s generation), producing correct JSON-only function calls.

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
