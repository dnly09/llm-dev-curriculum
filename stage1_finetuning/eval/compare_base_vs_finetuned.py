"""
Stage 1 — qualitative eval: base model vs fine-tuned (LoRA) model, same prompts,
read side by side. This is the step that actually tells you whether the fine-tune
helped, as opposed to loss curves which only tell you training was numerically healthy.

Usage:
    python compare_base_vs_finetuned.py
"""
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
ADAPTER_PATH = "../unsloth/lora_finetome"   # adjust if you run this from elsewhere
MAX_NEW_TOKENS = 200

# A handful of held-out prompts — NOT from the training set. Mix general
# instruction-following styles since this was a general (FineTome) fine-tune.
PROMPTS = [
    "Explain the difference between a list and a tuple in Python.",
    "Write a short professional email declining a meeting invitation due to a scheduling conflict.",
    "What are three tips for improving focus while studying?",
    "Summarize the plot of Romeo and Juliet in two sentences.",
    "How does a refrigerator keep food cold? Explain simply.",
]


def generate(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    full = tokenizer.decode(out[0], skip_special_tokens=True)
    # strip the prompt echo, keep only the generated continuation
    return full[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()


def main():
    print("Loading base model...")
    base_model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048, load_in_4bit=True, dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(base_model)

    print("Generating base model outputs...\n")
    base_outputs = [generate(base_model, tokenizer, p) for p in PROMPTS]

    # Free the base model before loading the fine-tuned one to keep VRAM low
    del base_model
    import torch, gc
    gc.collect()
    torch.cuda.empty_cache()

    print("Loading fine-tuned (LoRA) model...")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER_PATH, max_seq_length=2048, load_in_4bit=True, dtype=None,
    )
    ft_tokenizer = get_chat_template(ft_tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(ft_model)

    print("Generating fine-tuned model outputs...\n")
    ft_outputs = [generate(ft_model, ft_tokenizer, p) for p in PROMPTS]

    # Print side by side
    for i, prompt in enumerate(PROMPTS):
        print("=" * 100)
        print(f"PROMPT {i+1}: {prompt}")
        print("-" * 100)
        print(f"[BASE]\n{base_outputs[i]}\n")
        print("-" * 100)
        print(f"[FINE-TUNED]\n{ft_outputs[i]}\n")

    print("=" * 100)
    print("\nRead each pair and judge for yourself: more helpful, better formatted,")
    print("more concise, closer to how you'd want an assistant to respond? That's")
    print("the real signal — the loss numbers only told you training was stable.")


if __name__ == "__main__":
    main()