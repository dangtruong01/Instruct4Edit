"""
Qwen2.5-VL-7B Fine-tuning Script for Vision-Language Tasks

Prerequisites:
1. Install the required packages:
   pip install torch torchvision accelerate peft bitsandbytes datasets pillow
   
2. Install transformers from source (required for Qwen2.5-VL):
   pip install git+https://github.com/huggingface/transformers
   
3. Install qwen-vl-utils:
   pip install qwen-vl-utils[decord]
   # or for non-Linux systems:
   pip install qwen-vl-utils

4. Ensure your data structure matches:
   - vl_train.json with entries containing:
     - "id": unique identifier
     - "instruction": text instruction
     - "original_image": path to image file
     - "original_html": original HTML code
     - "modified_html": target HTML output
     - "modified_image": (optional) path to modified image
"""

import json
import torch
import os
import warnings
from datasets import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from qwen_vl_utils import process_vision_info

# Suppress warnings
warnings.filterwarnings("ignore")

# Environment setup
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def process_func(sample, processor):
    """
    Process function adapted from train.py for VL tasks
    Uses the same prompt structure but includes image input
    This function is called lazily during training, not upfront
    """
    # Load image lazily when needed
    try:
        image = Image.open(sample['original_image']).convert('RGB')
    except Exception as e:
        print(f"Error loading image {sample['original_image']}: {e}")
        # Create a blank image as fallback
        image = Image.new('RGB', (1280, 720), color='white')
    
    # Create conversation using the exact prompt from train.py but with image
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "You are a frontend developer. Given the HTML code and a design instruction, "
                    "apply the change and return the full modified HTML file. "
                    "Do NOT skip any parts of the code — output the complete modified HTML with only the necessary edits.\n\n"
                    f"Instruction:\n{sample['instruction']}\n\n"
                    f"Original HTML:\n{sample['original_html']}\n\n"
                    "Modified HTML:"
                )}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample['modified_html']}
            ]
        }
    ]
    
    # Apply chat template to get the formatted prompt
    text = processor.apply_chat_template(
        conversations, 
        tokenize=False, 
        add_generation_prompt=False
    )
    
    # Extract images and videos using qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(conversations)
    
    # Process the inputs through the processor
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=True,
        max_length=8192,
        return_tensors="pt"
    )
    
    # Create labels for training (copy input_ids)
    labels = inputs["input_ids"].clone()
    
    # Find where the assistant starts responding
    assistant_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    input_ids = inputs["input_ids"].squeeze()
    
    # Mask out everything except the assistant's response
    mask_until = 0
    for i in range(len(input_ids) - 1):
        if input_ids[i] == assistant_token_id:
            assistant_text = processor.tokenizer.decode(input_ids[i:i+10], skip_special_tokens=False)
            if "assistant" in assistant_text:
                assistant_end = i + 3
                mask_until = assistant_end
                break
    
    # Mask labels before assistant response
    labels[:, :mask_until] = -100
    
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "pixel_values": inputs.get("pixel_values", torch.empty(0)).squeeze() if inputs.get("pixel_values") is not None else None,
        "image_grid_thw": inputs.get("image_grid_thw", torch.empty(0)).squeeze() if inputs.get("image_grid_thw") is not None else None,
        "labels": labels.squeeze()
    }

class QwenVLDataCollator:
    """Custom data collator for handling vision-language inputs"""
    
    def __init__(self, processor):
        self.processor = processor
    
    def __call__(self, features):
        # Process features lazily here - features come from the dataset map function
        processed_features = []
        for feature in features:
            # If feature is not yet processed (contains raw data), process it now
            if 'original_image' in feature:
                processed_feature = process_func(feature, self.processor)
                processed_features.append(processed_feature)
            else:
                # Already processed
                processed_features.append(feature)
        
        # Separate different types of inputs
        input_ids = [f["input_ids"] for f in processed_features]
        attention_masks = [f["attention_mask"] for f in processed_features]
        labels = [f["labels"] for f in processed_features]
        
        # Pad sequences
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100
        )
        
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels
        }
        
        # Handle pixel values if present
        pixel_values = [f["pixel_values"] for f in processed_features if f["pixel_values"] is not None]
        if pixel_values and len(pixel_values) == len(processed_features):
            batch["pixel_values"] = torch.stack(pixel_values)
        
        # Handle image grid info if present
        image_grid_thw = [f["image_grid_thw"] for f in processed_features if f["image_grid_thw"] is not None]
        if image_grid_thw and len(image_grid_thw) == len(processed_features):
            batch["image_grid_thw"] = torch.stack(image_grid_thw)
        
        return batch

def main():
    print("Starting Qwen2.5-VL fine-tuning...")
    
    # 1. Load dataset (following train.py pattern - only load JSON, don't process yet)
    print("Loading dataset...")
    with open("data/vl_train_unfiltered_original_only.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    print(f"Loaded {len(raw_data)} training samples")
    
    # 2. Load model and processor (adapted from train.py style)
    model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        use_fast=True
    )
    
    # Set pad token if not already set (from train.py)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # attn_implementation="flash_attention_2"
    )
    
    # 3. Model configuration (following train.py)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    # 4. Create dataset from raw data (no preprocessing yet - following train.py lazy approach)
    print("Creating dataset...")
    dataset = Dataset.from_list(raw_data)
    
    # 5. LoRA Configuration (same as train.py)
    print("Setting up LoRA...")
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
    peft_model.print_trainable_parameters()
    
    # 6. Training arguments (combining train.py approach with VL adjustments)
    training_args = TrainingArguments(
        output_dir="./models/qwen2.5-vl-7b-finetuned-design-edit",
        per_device_train_batch_size=1,          # Small for VL models
        gradient_accumulation_steps=8,          # From train.py
        num_train_epochs=3,                     # From train.py
        learning_rate=2e-5,                     # From train.py
        lr_scheduler_type="cosine",             # From train.py
        warmup_ratio=0.1,                       # From train.py
        bf16=True,                              # From train.py
        save_strategy="epoch",                  # From train.py
        save_total_limit=2,                     # From train.py
        report_to=None,                         # From train.py
        logging_steps=10,                       # From train.py
        remove_unused_columns=False,            # Important for VL models
        dataloader_num_workers=0,               # From train.py (optimized)
        optim="adamw_bnb_8bit",                # From train.py
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        dataloader_pin_memory=False,            # VL-specific
    )
    
    # 7. Data collator (VL-specific - now handles lazy processing)
    data_collator = QwenVLDataCollator(processor)
    
    # 8. Trainer (following train.py pattern)
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    # 9. Memory management and training (from train.py)
    print(f"Dataset size: {len(dataset)}")
    print("Memory summary before training:")
    print(torch.cuda.memory_summary())
    torch.cuda.empty_cache()
    print("Memory summary after empty_cache:")
    print(torch.cuda.memory_summary())
    
    print("Starting training...")
    trainer.train()
    print("Training finished.")
    
    # 10. Save model (following train.py pattern)
    output_dir = "./models/qwen2.5-vl-7b-finetuned-design-edit"
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    
    print(f"Training completed! Model saved to {output_dir}")

if __name__ == "__main__":
    main()