"""
Stage 2, Track A — from-scratch KD loss (understand it once).

Goal: one teacher forward pass, one student forward pass, one optimizer step,
using the real Qwen2.5-0.5B (student) / Qwen2.5-7B-Instruct (teacher, 4-bit) pair.
Point is to *watch* what temperature (T) and alpha do to the loss, not to train
anything useful yet — Tiers 1-3 (framework-based) come after this.

Run inside .venv-distill:
    python kd_loss_scratch.py
"""
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

STUDENT_ID = "Qwen/Qwen2.5-0.5B"
TEACHER_ID = "Qwen/Qwen2.5-7B-Instruct"

# A couple of short function-calling-flavored prompts (swap in real glaive-fc rows later)
PROMPTS = [
    "User: What's the weather in Boston?\nAssistant:",
    "User: Convert 5 kilometers to miles.\nAssistant:",
]


def kd_loss(student_logits, teacher_logits, labels, T=2.0, alpha=0.5):
    """
    student_logits, teacher_logits: [B, S, V]  (already shifted for next-token)
    labels: [B, S] with -100 on prompt/pad positions
    T: temperature (softens distributions). alpha: weight on soft (KD) vs hard (CE).
    """
    s_logp = F.log_softmax(student_logits / T, dim=-1)
    t_prob = F.softmax(teacher_logits / T, dim=-1)
    kd = F.kl_div(s_logp, t_prob, reduction="none").sum(-1)      # [B, S]
    mask = (labels != -100)
    kd = (kd * mask).sum() / mask.sum() * (T * T)                # T^2 keeps grads scaled

    ce = F.cross_entropy(
        student_logits.flatten(0, 1),
        labels.flatten(),
        ignore_index=-100,
    )

    return alpha * kd + (1.0 - alpha) * ce, kd.item(), ce.item()


def main():
    print("Loading tokenizer (student's — shared Qwen family vocab)...")
    tok = AutoTokenizer.from_pretrained(STUDENT_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading student (bf16, full precision — it's small)...")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID, torch_dtype=torch.bfloat16
    ).to("cuda")

    print("Loading teacher (4-bit NF4 — frozen, eval mode)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_ID, quantization_config=bnb_config, torch_dtype=torch.bfloat16
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # --- tokenize a tiny batch ---
    enc = tok(PROMPTS, return_tensors="pt", padding=True).to("cuda")
    input_ids = enc["input_ids"]
    attn_mask = enc["attention_mask"]

    # For this smoke exercise, "labels" = input_ids shifted, with pad masked to -100.
    # (In the real training loop, labels come from your data collator and only
    # cover completion tokens — prompt tokens are -100 there too.)
    labels = input_ids.clone()
    labels[attn_mask == 0] = -100

    # --- forward passes ---
    print("Teacher forward pass (frozen, no_grad)...")
    with torch.no_grad():
        teacher_out = teacher(input_ids=input_ids, attention_mask=attn_mask)
    teacher_logits = teacher_out.logits.to("cuda").float()

    print("Student forward pass...")
    student_out = student(input_ids=input_ids, attention_mask=attn_mask)
    student_logits = student_out.logits.float()

    # --- truncate to the REAL tokenizer vocab size ---
    # Qwen2.5 checkpoints pad the lm_head/embedding matrix to different widths
    # depending on model size (tensor-core efficiency), independent of
    # base-vs-instruct: e.g. 0.5B pads to 151936, 7B pads to 152064, even though
    # the underlying tokenizer vocab is identical and smaller than both. Those
    # extra columns are untrained padding — slice them off before comparing
    # distributions, or the KL term compares noise.
    true_vocab = len(tok)
    print(f"student config vocab_size: {student.config.vocab_size}")
    print(f"teacher config vocab_size: {teacher.config.vocab_size}")
    print(f"tokenizer true vocab_size (len(tok)): {true_vocab}")
    student_logits = student_logits[..., :true_vocab]
    teacher_logits = teacher_logits[..., :true_vocab]

    # --- shift for next-token prediction ---
    s_logits_shifted = student_logits[:, :-1, :]
    t_logits_shifted = teacher_logits[:, :-1, :]
    labels_shifted = labels[:, 1:]

    optimizer = torch.optim.AdamW(student.parameters(), lr=2e-5)

    # --- Part 1: SWEEP (observation only, no backward/step) ---
    # We want to *watch* how T and alpha move the loss components on the exact
    # same forward pass. Calling .backward() here would (a) crash on the 2nd
    # combo, since PyTorch frees intermediate activations after the first
    # .backward(), and (b) even with retain_graph=True, an .optimizer.step()
    # between combos would silently change the weights mid-sweep, so later
    # combos wouldn't be comparing like-for-like anymore. Detach + no_grad
    # keeps this a clean, repeatable inspection.
    print("\n--- Sweeping T and alpha (read-only — same forward pass, no training) ---")
    s_detached = s_logits_shifted.detach()
    t_detached = t_logits_shifted.detach()
    with torch.no_grad():
        for T in (1.0, 2.0, 4.0):
            for alpha in (0.0, 0.5, 0.9):
                loss, kd_val, ce_val = kd_loss(
                    s_detached, t_detached, labels_shifted, T=T, alpha=alpha
                )
                print(
                    f"T={T:>4.1f}  alpha={alpha:>4.2f}  "
                    f"total={loss.item():.4f}  kd_term={kd_val:.4f}  ce_term={ce_val:.4f}"
                )

    # --- Part 2: ONE real training step (guide's actual advance criterion) ---
    # T=2.0, alpha=0.5 are the guide's suggested starting values (§4). This is
    # the graph that's allowed to backprop — it's fresh, and we only call
    # .backward()/.step() once, so there's no freed-graph or stale-weights issue.
    print("\n--- One real training step (T=2.0, alpha=0.5) ---")
    optimizer.zero_grad()
    loss, kd_val, ce_val = kd_loss(
        s_logits_shifted, t_logits_shifted, labels_shifted, T=2.0, alpha=0.5
    )
    loss.backward()
    optimizer.step()
    print(
        f"total={loss.item():.4f}  kd_term={kd_val:.4f}  ce_term={ce_val:.4f}  "
        "-> optimizer.step() applied to student"
    )

    print("\nKD LOSS FROM-SCRATCH SMOKE TEST DONE ✅")
    print(
        "Read the sweep table: as T rises, kd_term should shrink relative to a fixed "
        "alpha (softer distributions -> smaller KL in absolute terms before the T^2 "
        "rescale kicks in); as alpha rises, total should lean further toward kd_term "
        "and away from ce_term."
    )


if __name__ == "__main__":
    main()