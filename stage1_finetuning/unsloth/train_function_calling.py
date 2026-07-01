"""
Stage 1 — the judgeable run: QLoRA fine-tune for function/tool calling.

Unlike the FineTome run, this task has a real exploitable gap: the base model
can attempt function calls, but is inconsistent about matching a specific JSON
format. A small adapter is good at enforcing rigid format compliance -- so we
expect a much clearer before/after than general chat gave us.

Dataset: hiyouga/glaive-function-calling-v2-sharegpt
    - "tools": JSON string, list of available function schemas for this example
    - "conversations": list of {"from": "human"|"gpt"|"function_call"|"observation",
      "value": ...}. function_call values are already valid JSON: {"name":..., "arguments":{...}}

We keep only examples where at least one function is available (non-empty "tools"),
build a system prompt from the schema, and map roles onto a plain user/assistant
chat format (function_call -> assistant turn containing the JSON; observation ->
a user turn prefixed "FUNCTION RESPONSE:", matching the original dataset's convention).

Usage:
    python train_function_calling.py
"""
import json

from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
MAX_SEQ = 2048
OUTPUT_DIR = "outputs_fc"
N_EXAMPLES = 3000   # more than the FineTome run -- format learning benefits from more examples

SYSTEM_TEMPLATE = (
    "You are a helpful assistant with access to the following functions. "
    "Use them if required.\n\n"
    "If you call a function, respond with ONLY a JSON object of the form "
    '{{"name": "<function name>", "arguments": {{...}}}}, and nothing else -- '
    "no extra text, no markdown formatting.\n\n"
    "Available functions:\n{tools_json}"
)

ROLE_MAP = {"human": "user", "gpt": "assistant"}   # function_call/observation handled specially


def build_messages(example):
    tools = json.loads(example["tools"]) if example["tools"] else []
    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(
        tools_json=json.dumps(tools, indent=2))}]
    for turn in example["conversations"]:
        role, value = turn["from"], turn["value"]
        if role == "function_call":
            messages.append({"role": "assistant", "content": value})
        elif role == "observation":
            messages.append({"role": "user", "content": f"FUNCTION RESPONSE: {value}"})
        else:
            messages.append({"role": ROLE_MAP[role], "content": value})
    return messages


def main():
    print("Loading base in 4-bit...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=MAX_SEQ, load_in_4bit=True, dtype=None,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("Loading and filtering dataset...")
    ds = load_dataset("hiyouga/glaive-function-calling-v2-sharegpt", split="train")
    ds = ds.filter(lambda ex: ex["tools"] and ex["tools"] != "[]")
    ds = ds.shuffle(seed=3407).select(range(min(N_EXAMPLES, len(ds))))
    print(f"Examples with available tools: {len(ds)}")

    def to_text(ex):
        messages = build_messages(ex)
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)}

    ds = ds.map(to_text)

    split = ds.train_test_split(test_size=0.08, seed=3407)
    train_ds, eval_ds = split["train"], split["test"]
    print(f"Train: {len(train_ds)} | Eval (held out): {len(eval_ds)}")

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_ds, eval_dataset=eval_ds,
        args=SFTConfig(
            dataset_text_field="text", max_length=MAX_SEQ,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=2,
            eval_strategy="steps",
            eval_steps=30,
            warmup_steps=10,
            num_train_epochs=1,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=OUTPUT_DIR,
            report_to="none",
        ),
    )

    train_result = trainer.train()
    print("\nFinal train loss:", train_result.training_loss)

    eval_result = trainer.evaluate()
    print("Final eval loss:", eval_result["eval_loss"])

    model.save_pretrained("lora_function_calling")
    tokenizer.save_pretrained("lora_function_calling")
    print("\nSaved LoRA adapters to ./lora_function_calling")
    print("Next: run eval/score_function_calling.py for pass/fail scoring vs. base model")


if __name__ == "__main__":
    main()