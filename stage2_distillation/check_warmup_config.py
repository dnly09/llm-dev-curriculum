"""
One-shot diagnostic for the warmup_steps discrepancy in the Tier 3 full run.
No GPU/model loading required -- just constructs the same GKDConfig and
checks what it resolves to, to isolate "config problem" from "trainer
runtime problem" without spending another 2-hour run to find out.

Usage:
    python check_warmup_config.py
"""
from trl.experimental.gkd import GKDConfig

cfg = GKDConfig(
    output_dir="warmup_check_scratch",
    warmup_steps=30,
    num_train_epochs=1,
)

print("cfg.warmup_steps:", cfg.warmup_steps)
print("cfg.warmup_ratio:", cfg.warmup_ratio)
print("cfg.get_warmup_steps(345):", cfg.get_warmup_steps(345))
print("cfg.lr_scheduler_type:", cfg.lr_scheduler_type)