import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers.utils import logging
import os

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Logging
logging.set_verbosity_info()

# 1. Preprocessing function
def process_func(example, tokenizer):
    prompt = (
        "You are a frontend developer. Given the HTML code and a design instruction, "
        "apply the change and return the full modified HTML file. "
        "Do NOT skip any parts of the code — output the complete modified HTML with only the necessary edits.\n\n"
        f"Instruction:\n{example['instruction']}\n\n"
        f"Original HTML:\n{example['original_html']}\n\n"
        "Modified HTML:"
    )
    target = example["modified_html"]

    input_enc = tokenizer(prompt, add_special_tokens=True)
    target_enc = tokenizer(target, add_special_tokens=False)

    input_ids = input_enc["input_ids"] + target_enc["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = input_enc["attention_mask"] + [1] * (len(target_enc["input_ids"]) + 1)
    labels = [-100] * len(input_enc["input_ids"]) + target_enc["input_ids"] + [tokenizer.eos_token_id]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 2. Load data
with open("data/train.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 3. Load tokenizer and model (non-VL)
model_name = "/root/Qwen2.5-7B-Instruct-1M"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
)
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# 4. Preprocess data
processed_data = [process_func(sample, tokenizer) for sample in raw_data]
dataset = Dataset.from_list(processed_data)

# 5. LoRA Configuration
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
)
peft_model = get_peft_model(model, config)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./models/qwen2.5-7b-instruct-1m-finetuned-design-edit",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    bf16=True,
    save_strategy="epoch",
    save_total_limit=2,
    report_to=None,
    logging_steps=10,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    optim="adamw_bnb_8bit",
)

# 7. Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=peft_model, padding=True)

# 8. Trainer
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

# 9. Train
print(f"Dataset size: {len(dataset)}")
print("Memory summary before training:")
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
print("Memory summary after empty_cache:")
print(torch.cuda.memory_summary())

print("Starting training...")
# trainer.train()
trainer.train(resume_from_checkpoint="./models/qwen2.5-7b-instruct-1m-finetuned-design-edit/checkpoint-390")
print("Training finished.")

# 10. Save model
output_dir = "./models/qwen2.5-7b-instruct-1m-finetuned-design-edit"
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
