"""
Tier 3 — GKDTrainer subclass with a vocab-width truncation fix.

Why this exists (see PROGRESS.md / README_stage2_v2.md "known gotcha"):
Qwen2.5 checkpoints pad their lm_head/embedding matrix to a width that
depends on MODEL SIZE, not on the real tokenizer vocab:
    Qwen2.5-0.5B-Instruct lm_head width: 151936
    Qwen2.5-7B-Instruct   lm_head width: 152064
    real vocab (len(tokenizer)):         ~151665, identical across the family

Stock trl.experimental.gkd.GKDTrainer.generalized_jsd_loss takes
student_logits / teacher_logits straight from each model's `.logits`
output with no truncation (verified against the installed trl==1.7.0
source). Both the beta=0/1 F.kl_div path and the 0<beta<1 torch.stack
mixture path require identical last-dim shape, so with this teacher/
student pair the trainer would either raise a shape-mismatch error, or
compare misaligned logit columns if a shape ever coincidentally matched.

Fix: truncate both logit tensors to `len(tokenizer)` (the real, shared
vocab) before the JSD call — same principle as capture_teacher_logits.py
in Tier 2. We only override the non-Liger `compute_loss` branch (the
Liger fused-kernel path is a separate code path this project isn't using
for Tier 3; if that changes later, it needs the same truncation applied
to `student_head.weight` / `teacher_head.weight` before the fused call).

Everything else (training_step's on-policy generation branch, the ChatML
collator, teacher loading via GKDConfig.teacher_model_init_kwargs) is
inherited unchanged from GKDTrainer.
"""

from trl.experimental.gkd import GKDTrainer
from trl.experimental.utils import empty_cache


class GKDTrainerTruncated(GKDTrainer):
    """GKDTrainer with student/teacher logits truncated to the shared
    real vocab size before computing the generalized JSD loss.

    Use this whenever teacher and student are different-size checkpoints
    from the same model family whose lm_head padding widths differ (as
    with Qwen2.5-0.5B vs Qwen2.5-7B). If you ever pair same-size models
    (identical lm_head width), this subclass is a no-op and still safe.
    """

    def __init__(self, *args, vocab_size: int, **kwargs):
        """
        Args:
            vocab_size: the real, shared vocab size to truncate both
                student and teacher logits to. Pass `len(tokenizer)`
                — NOT `tokenizer.vocab_size` (that excludes added
                special tokens on some tokenizers) and NOT either
                model's `lm_head` width. Compute this once against the
                *student* tokenizer before constructing the trainer,
                since GKDConfig's `processing_class` is the student's.
        """
        super().__init__(*args, **kwargs)
        self._gkd_vocab_size = vocab_size

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_liger_gkd_loss:
            # Not truncated — this project doesn't enable use_liger_kernel for
            # Tier 3. Raise loudly instead of silently comparing padded-noise
            # columns if someone flips that config flag later.
            raise NotImplementedError(
                "GKDTrainerTruncated does not yet truncate the Liger fused-JSD "
                "path (student_head.weight / teacher_head.weight). Either keep "
                "GKDConfig(use_liger_kernel=False), or extend this method to "
                "slice both head weight matrices to self._gkd_vocab_size before "
                "the LigerFusedLinearJSDLoss call."
            )

        # compute student output
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

        # compute teacher output in eval mode
        self.teacher_model.eval()
        import torch

        with torch.no_grad():
            teacher_outputs = self.teacher_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        # Standard causal shift, matching the parent implementation exactly.
        shifted_student_logits = student_outputs.logits[:, :-1, :]
        shifted_teacher_logits = teacher_outputs.logits[:, :-1, :]
        shifted_labels = inputs["labels"][:, 1:]

        # --- the fix: truncate both to the real, shared vocab width ---
        v = self._gkd_vocab_size
        shifted_student_logits = shifted_student_logits[..., :v]
        shifted_teacher_logits = shifted_teacher_logits[..., :v]
        # ---------------------------------------------------------------

        loss = self.generalized_jsd_loss(
            student_logits=shifted_student_logits,
            teacher_logits=shifted_teacher_logits,
            labels=shifted_labels,
            beta=self.beta,
            num_items_in_batch=num_items_in_batch,
        )

        empty_cache()
        return (loss, student_outputs) if return_outputs else loss