"""
Stage 1, Track A — LoRA from scratch.

The whole idea:
    h = W x + (alpha / r) * (B @ (A @ x))
    #   frozen              trainable: A is r x d_in, B is d_out x r

Run this file directly for a quick CPU sanity check (no GPU, no model download):
    python lora_linear.py
"""
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank additive update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():   # freeze W and bias
            p.requires_grad_(False)

        d_out, d_in = base.weight.shape
        # A: small random init (so the update direction isn't zero from the start)
        self.A = nn.Parameter(torch.randn(r, d_in) * (1 / r ** 0.5))
        # B: zero init => at step 0, B @ A = 0 => adapter is a no-op initially
        self.B = nn.Parameter(torch.zeros(d_out, r))
        self.scale = alpha / r

    def forward(self, x):
        base_out = self.base(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + self.scale * lora_out

    def trainable_parameters(self):
        return [self.A, self.B]


if __name__ == "__main__":
    torch.manual_seed(0)

    # --- Sanity check 1: at init, LoRA is a no-op ---
    base = nn.Linear(32, 16, bias=False)
    lora = LoRALinear(base, r=4, alpha=8)
    x = torch.randn(2, 32)
    with torch.no_grad():
        out_base = base(x)
        out_lora = lora(x)
    assert torch.allclose(out_base, out_lora), "LoRA should be a no-op at init (B=0)!"
    print("✅ At init, LoRALinear output == base output (B starts at zero)")

    # --- Sanity check 2: only A and B receive gradients, W stays frozen ---
    target = torch.randn(2, 16)
    out = lora(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    assert lora.base.weight.grad is None, "Base weight should be frozen (no grad)!"
    assert lora.A.grad is not None and lora.B.grad is not None, "A and B should have grads!"
    print("✅ Base weight frozen (no grad); A and B received gradients")

    # --- Sanity check 3: parameter count matches the "small fraction" claim ---
    base_params = sum(p.numel() for p in base.parameters())
    lora_params = sum(p.numel() for p in lora.trainable_parameters())
    print(f"Base params: {base_params} | Trainable LoRA params: {lora_params} "
          f"({100 * lora_params / base_params:.1f}% of base)")

    print("\nAll checks passed. The mechanism is doing exactly what the math says.")