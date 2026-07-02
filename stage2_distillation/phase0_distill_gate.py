# phase0_distill_gate.py
import transformers, trl, torch
print("transformers:", transformers.__version__)
print("trl:", trl.__version__)

try:
    from trl.experimental.gkd import GKDConfig, GKDTrainer
    print("GKD import: trl.experimental.gkd ✅")
except ImportError:
    from trl import GKDConfig, GKDTrainer
    print("GKD import: trl (stable) ✅")

import inspect
sig = inspect.signature(GKDConfig.__init__)
print("has max_length:", "max_length" in sig.parameters,
      "| has max_seq_length:", "max_seq_length" in sig.parameters)
print("GKDConfig lmbda/beta/seq_kd present:",
      all(k in sig.parameters for k in ("lmbda","beta","seq_kd")))