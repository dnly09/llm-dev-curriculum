"""
Stage 2, Tier 2 — offline logit capture (no vLLM; plain transformers forward pass).

Design decisions (see session notes for reasoning):
  1. Target sequence = the ORIGINAL ground-truth conversation (system + user +
     function_call/assistant turn), NOT the teacher's Tier-1 generated completion.
     This is what makes it logit-level KD rather than "SFT on teacher text again" --
     both models are conditioned on the same real trajectory, and the teacher's
     full distribution at each position is the extra signal over the hard label.
  2. We add an explicit `labels` column (-100 on system/user/pad positions) to the
     captured schema, because DistillationTrainer.compute_loss() defaults to
     labels = input_ids (no masking) whenever no `labels` key is present in the batch.
     Without this, the student would be supervised on prompt tokens too.
  3. "Filtered" reuses Tier 1's exact pass/fail classification (from
     filter_teacher_completions.py, applied to the already-saved
     teacher_completions.jsonl) to select which of the 2760 ORIGINAL examples to
     keep -- not a re-filter of generated text, since we no longer train on that text.
  4. Compression config starts lossless-ish: k == exact_k (top-k stored exactly,
     no polynomial/residual approximation). Storage isn't a constraint at this
     scale (~2700 examples); simplicity first, revisit compression later if desired.
  5. Fixed-length, right-padded sequences (prepacked=True downstream) -- DistillKit's
     prepacked loader requires uniform length across rows.

Usage:
    python capture_teacher_logits.py --split unfiltered --output ./teacher_logits_unfiltered/
    python capture_teacher_logits.py --split filtered   --output ./teacher_logits_filtered/
"""

import json
import re
import argparse

import torch
import pyarrow
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from distillkit.compression import LogprobCompressor
from distillkit.compression.config import DistributionQuantizationConfig
from distillkit.sample_common import StreamingParquetWriter

TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"
COMPLETIONS_PATH = "teacher_completions.jsonl"

# --- must match generate_teacher_completions.py / filter_teacher_completions.py exactly ---
N_EXAMPLES = 3000
SPLIT_SEED = 3407
TEST_SIZE = 0.08

MAX_SEQ_LEN = 768  # function-calling prompts + short completions comfortably fit
TOP_K = 64  # k == exact_k -> no lossy compression to start

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)

ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "function_call": "assistant",
    "function_response": "tool",
    "system": "system",
}


def build_train_split():
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=SPLIT_SEED).select(range(min(N_EXAMPLES, len(ds))))
    split = ds.train_test_split(test_size=TEST_SIZE, seed=SPLIT_SEED)
    return split["train"]


def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def get_filtered_indices(train_ds):
    """Reproduce filter_teacher_completions.py's pass/fail classification exactly,
    against the already-saved teacher_completions.jsonl. Returns the set of
    train_ds indices that passed (matching the 2157-example filtered set)."""
    completions = [json.loads(l) for l in open(COMPLETIONS_PATH)]
    assert len(train_ds) == len(
        completions
    ), "ordering assumption broken -- do not proceed"

    keep_idx = []
    for i, (ex, row) in enumerate(zip(train_ds, completions)):
        convo = ex["conversations"]
        expects_fc = len(convo) >= 2 and convo[1]["from"] == "function_call"
        parsed = extract_json(row["teacher_completion"])
        teacher_emitted_json = parsed is not None

        if expects_fc:
            if teacher_emitted_json and isinstance(parsed, dict) and "name" in parsed:
                keep_idx.append(i)
        else:
            if not teacher_emitted_json:
                keep_idx.append(i)
    return keep_idx


def build_example_messages(ex):
    """Full ground-truth conversation: system (tools) + human + the real next turn.
    Only the first 3 messages are ever used downstream -- don't inspect anything
    beyond that, so later multi-turn roles (e.g. 'observation') never matter."""
    tools = json.loads(ex["tools"]) if ex["tools"] else []
    system = SYSTEM_TEMPLATE.format(tools_json=json.dumps(tools, indent=2))
    convo = ex["conversations"]
    if not convo or convo[0]["from"] != "human" or len(convo) < 2:
        return None

    messages = [{"role": "system", "content": system}]
    for turn in convo[:2]:  # only human + its immediate response
        role = ROLE_MAP.get(turn["from"])
        if role is None:
            return None
        if turn["from"] == "function_call":
            messages.append({"role": "assistant", "content": turn["value"]})
        else:
            messages.append({"role": role, "content": turn["value"]})
    return messages


def tokenize_with_labels(messages, tokenizer, max_seq_len):
    """Tokenize full conversation; mask everything except the FIRST assistant
    turn's tokens (the target the student should learn -- matches the single-turn
    scope generate_teacher_completions.py used for the teacher's own generation)."""
    # tokens for prompt only (system + human), no generation prompt marker needed
    # since we're supplying the real completion ourselves
    prompt_msgs = messages[:2]  # [system, human]
    assert prompt_msgs[0]["role"] == "system" and prompt_msgs[1]["role"] == "user"
    full_msgs = messages[:3]  # [system, human, first assistant/function_call turn]
    if len(full_msgs) < 3:
        return None

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    if not full_text.startswith(prompt_text):
        # chat template inserted something between prompt and completion that
        # breaks the simple prefix assumption -- skip rather than silently misalign
        return None

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_seq_len or len(prompt_ids) >= len(full_ids):
        return None

    pad_id = tokenizer.pad_token_id
    input_ids = full_ids + [pad_id] * (max_seq_len - len(full_ids))
    labels = (
        [-100] * len(prompt_ids)
        + full_ids[len(prompt_ids) :]
        + [-100] * (max_seq_len - len(full_ids))
    )
    return input_ids, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["filtered", "unfiltered"], required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    print("Loading teacher tokenizer + model (4-bit NF4)...")
    tok = AutoTokenizer.from_pretrained(TEACHER_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # gotcha #1: real vocab (~151665) vs padded lm_head (151936 for 0.5B / 152064 for 7B)
    real_vocab_size = max(len(tok.get_vocab()), max(tok.get_vocab().values()) + 1)
    print(f"Real tokenizer vocab size (truncation target): {real_vocab_size}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, quantization_config=bnb_config, dtype=torch.bfloat16
    )
    teacher.eval()

    train_ds = build_train_split()
    if args.split == "filtered":
        keep_idx = set(get_filtered_indices(train_ds))
        train_ds = train_ds.select(sorted(keep_idx))
        print(f"Filtered split: {len(train_ds)} examples")
    else:
        print(f"Unfiltered split: {len(train_ds)} examples")

    comp_config = DistributionQuantizationConfig(
        d=real_vocab_size,
        k=TOP_K,
        exact_k=TOP_K,  # k == exact_k -> lossless top-k, no polynomial approx
        residual_bins=[],
        polynomial_terms=None,
    )
    compressor = LogprobCompressor(config=comp_config)

    schema = pyarrow.schema(
        [
            pyarrow.field("input_ids", pyarrow.list_(pyarrow.uint64())),
            pyarrow.field("labels", pyarrow.list_(pyarrow.int64())),
            pyarrow.field(
                "compressed_logprobs", pyarrow.list_(pyarrow.list_(pyarrow.uint8()))
            ),
            pyarrow.field(
                "bytepacked_indices", pyarrow.list_(pyarrow.list_(pyarrow.uint8()))
            ),
        ]
    )

    writer = StreamingParquetWriter(args.output, schema, file_max_rows=5000, queue_maxsize=0)
    writer.start()

    # save the compression config alongside the data -- the training YAML must
    # reference identical fields (d/k/exact_k/etc.) or decompression breaks
    with open(f"{args.output.rstrip('/')}_compression_config.json", "w") as f:
        json.dump(comp_config.model_dump(mode="json"), f, indent=2)

    n_written, n_skipped = 0, 0
    for ex in train_ds:
        messages = build_example_messages(ex)
        if messages is None:
            n_skipped += 1
            continue
        result = tokenize_with_labels(messages, tok, MAX_SEQ_LEN)
        if result is None:
            n_skipped += 1
            continue
        input_ids, labels = result

        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], device=teacher.device)
            logits = teacher(input_ids=input_tensor).logits[0, :, :real_vocab_size]
            logprobs = torch.log_softmax(logits.float(), dim=-1)

        compressed = compressor.compress(logprobs)  # dict of tensors, shape [seq, ...]

        writer.write(
            {
                "input_ids": input_ids,
                "labels": labels,
                "compressed_logprobs": compressed["compressed_logprobs"]
                .cpu()
                .numpy()
                .tolist(),
                "bytepacked_indices": compressed["bytepacked_indices"]
                .cpu()
                .numpy()
                .tolist(),
            }
        )
        n_written += 1
        if n_written % 200 == 0:
            print(f"  {n_written} captured...")

    writer.close()
    print(f"\nDone. Written: {n_written} | Skipped: {n_skipped}")
    print(f"Output: {args.output}")
    print(f"Compression config: {args.output.rstrip('/')}_compression_config.json")


if __name__ == "__main__":
    main()
