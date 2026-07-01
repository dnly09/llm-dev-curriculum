"""
Stage 1 — export the fine-tuned adapter for use in your existing llama.cpp stack.

Produces three artifacts (see guide §5):
    A) lora_model/          — adapters only, small, portable
    B) model_merged_16bit/  — standalone HF model (base + adapter merged)
    C) model_gguf/          — GGUF quantized for llama.cpp

Usage:
    python export_gguf.py
"""
from unsloth import FastLanguageModel

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
ADAPTER_PATH = "lora_finetome"   # adjust path if running from elsewhere
QUANT_METHOD = "q4_k_m"          # good default: quality/size tradeoff for llama.cpp

print("Loading base + adapter...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH, max_seq_length=2048, load_in_4bit=True, dtype=None,
)

# A) Adapters only (already saved during training, but re-saved here for completeness)
print("\nSaving adapters only...")
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")

# B) Merge to 16-bit — a standalone HF model, no longer needs the base + adapter split
print("\nMerging to 16-bit standalone model...")
model.save_pretrained_merged("model_merged_16bit", tokenizer, save_method="merged_16bit")

# C) GGUF for llama.cpp
print(f"\nExporting GGUF ({QUANT_METHOD})...")
model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method=QUANT_METHOD)

print("\nDone. GGUF file should be under ./model_gguf/")
print("Load it with your usual llama.cpp invocation, e.g.:")
print("  ~/llama.cpp/llama-cli -m model_gguf/*.gguf -p \"Explain the difference between a list and a tuple.\"")