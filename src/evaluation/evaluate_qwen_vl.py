import torch
import os
import json
import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from PIL import Image
from qwen_vl_utils import process_vision_info

# Paths - update to match your actual paths
base_model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
lora_weights_path = "./models/qwen2.5-vl-7b-finetuned-design-edit/checkpoint-256"
samples_dir = "/root/DesignEdit/evaluated_samples_qwen_vl"

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load processor (replaces tokenizer for VL models)
processor = AutoProcessor.from_pretrained(
    base_model_path,
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=1280 * 28 * 28
)

# Load base model on GPU
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for VL models
    trust_remote_code=True,
    device_map="auto"  # Auto distribute across available GPUs
)

# Load LoRA model
model = PeftModel.from_pretrained(base_model, lora_weights_path)
# print("Reducing overfitted LoRA influence...")
# for name, module in model.named_modules():
#     if hasattr(module, 'scaling'):
#         original = module.scaling
#         module.scaling = original * 0.5  # Reduce to 30% influence
#         print(f"Reduced {name}: {original} → {module.scaling}")

print("Merging LoRA weights...")
model = model.merge_and_unload()
model.eval()

class HTMLEndStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, input_ids, scores, **kwargs):
        # Don't check too early
        if input_ids.shape[1] < 100:
            return False
        
        # Only check the last 100 tokens to avoid stopping on original HTML
        last_tokens = self.tokenizer.decode(input_ids[0][-100:], skip_special_tokens=True)
        
        # Stop if we see </html> in the newly generated content
        if "</html>" in last_tokens.lower():
            # Make sure it's actually at the end (not in the middle)
            if last_tokens.strip().endswith("</html>") or "</html>" in last_tokens[-20:]:
                return True
        return False

def generate_output(instruction, original_html, original_image_path):
    """Generate model output for given instruction, HTML, and image - VERY HIGH LIMITS"""
    
    # Load image
    try:
        image = Image.open(original_image_path).convert('RGB')
        print(f"    Loaded image: {original_image_path}")
    except Exception as e:
        print(f"    Error loading image {original_image_path}: {e}")
        # Create a blank image as fallback
        image = Image.new('RGB', (1280, 720), color='white')
        print(f"    Using blank fallback image")
    
    # Create conversation using the exact same prompt as evaluate_qwen.py but with image
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": (
                    "You are a frontend developer. Given the HTML code, its rendered image, and a design edit instruction. "
                    "Read through the HTML code, and refer to the rendered image to understand the structure, styling, and components of the code."
                    "Apply the changes for the edit instruction and return the full modified HTML file. "
                    "Do NOT skip any parts of the code — output the complete modified HTML with only the necessary edits.\n\n"
                    f"Instruction:\n{instruction}\n\n"
                    f"Original HTML:\n{original_html}\n\n"
                    "Modified HTML:"
                )}
            ]
        }
    ]
    
    # Apply chat template to get the formatted prompt
    text = processor.apply_chat_template(
        conversations, 
        tokenize=False, 
        add_generation_prompt=True  # Set to True for generation
    )
    
    # Extract images and videos using qwen_vl_utils
    image_inputs, video_inputs = process_vision_info(conversations)
    
    # Process the inputs through the processor
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=False,
        truncation=False,
        return_tensors="pt"
    )
    
    # Move inputs to device
    inputs = {k: v.to(device) if v is not None else v for k, v in inputs.items()}
    
    print(f"    Input token count: {inputs['input_ids'].shape[1]}")
    
    # Add stopping criteria
    stopping_criteria = StoppingCriteriaList([HTMLEndStoppingCriteria(processor.tokenizer)])
    
    # Generate with VERY HIGH LIMITS for large HTML files
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=16384,       # Very high limit for large HTML files (16k tokens)
            do_sample=False,            # Greedy for consistency
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            early_stopping=False,       # Don't stop early
            use_cache=True,             # For efficiency
            repetition_penalty=1.02,    # Slight penalty to avoid loops
            stopping_criteria=stopping_criteria,  # Add stopping criteria
        )
        
        # Decode only the new tokens (exclude input)
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        output_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return output_text

def find_original_image(sample_path):
    """Find the original image file in the sample folder"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']
    
    for file in os.listdir(sample_path):
        if file.startswith('original') and any(file.lower().endswith(ext) for ext in image_extensions):
            return os.path.join(sample_path, file)
    
    # If no original image found, look for any image file
    for file in os.listdir(sample_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            return os.path.join(sample_path, file)
    
    return None

def process_sample_folder(sample_path):
    """Process a single sample folder"""
    print(f"Processing {sample_path}...")
    
    # Look for original HTML file
    original_html_path = None
    instruction_files = []
    
    for file in os.listdir(sample_path):
        if file.startswith('original') and file.endswith('.html'):
            original_html_path = os.path.join(sample_path, file)
        elif file.startswith('instruction_') and (file.endswith('.txt') or file.endswith('.json')):
            instruction_files.append(file)
    
    if not original_html_path:
        print(f"No original HTML file found in {sample_path}")
        return
    
    # Find original image
    original_image_path = find_original_image(sample_path)
    if not original_image_path:
        print(f"No original image file found in {sample_path}")
        return
    
    # Read original HTML
    with open(original_html_path, 'r', encoding='utf-8') as f:
        original_html = f.read()
    
    # Sort instruction files to ensure consistent ordering
    instruction_files.sort()
    
    # Process each instruction
    for i, inst_file in enumerate(instruction_files, 1):
        inst_path = os.path.join(sample_path, inst_file)
        
        # Read instruction
        with open(inst_path, 'r', encoding='utf-8') as f:
            if inst_file.endswith('.json'):
                instruction_data = json.load(f)
                instruction = instruction_data.get('instruction', instruction_data.get('text', str(instruction_data)))
            else:
                instruction = f.read().strip()
        
        print(f"  Processing instruction {i}: {instruction[:50]}...")
        print(f"    Original HTML length: {len(original_html)} characters")
        print(f"    Using image: {os.path.basename(original_image_path)}")
        
        # Generate output with timing
        start_time = time.time()
        try:
            output = generate_output(instruction, original_html, original_image_path)
        except Exception as e:
            print(f"    ERROR during generation: {e}")
            # Save error info and continue
            error_file = f"error_output_{i}.txt"
            error_path = os.path.join(sample_path, error_file)
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Error: {str(e)}\nInstruction: {instruction}\n")
            continue
            
        end_time = time.time()
        
        print(f"    Generation completed in {end_time - start_time:.2f} seconds")
        
        # Save output
        output_file = f"output_vl_{i}.html"
        output_path = os.path.join(sample_path, output_file)
        
        # The output should already be just the modified HTML since we're using add_generation_prompt=True
        modified_html = output.strip()
        
        # Remove markdown code fences if present
        if modified_html.startswith('```html'):
            modified_html = modified_html[7:].strip()
        if modified_html.endswith('```'):
            modified_html = modified_html[:-3].strip()
        
        # Quality checks
        print(f"    Generated HTML length: {len(modified_html)} characters")
        print(f"    Completion ratio: {len(modified_html)/len(original_html)*100:.1f}%")
        
        if modified_html.rstrip().endswith('</html>'):
            print("    ✓ HTML appears complete (ends with </html>)")
        else:
            print("    ⚠ WARNING: HTML may be incomplete")
            print(f"    Last 100 chars: ...{modified_html[-100:]}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(modified_html)
        
        print(f"  ✓ Saved output to {output_file}")
        
        # Save full raw output for debugging
        debug_file = f"debug_output_vl_{i}.txt"
        debug_path = os.path.join(sample_path, debug_file)
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(f"Full conversation output:\n{output}")
        print(f"  📄 Debug output saved to {debug_file}")
        
        # Memory cleanup
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            print(f"    GPU Memory after generation: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

def main():
    """Main function to process all sample folders"""
    if not os.path.exists(samples_dir):
        print(f"Samples directory {samples_dir} not found!")
        return
    
    # Get all sample folders
    sample_folders = [f for f in os.listdir(samples_dir) 
                     if os.path.isdir(os.path.join(samples_dir, f)) and f.startswith('sample_')]
    
    if not sample_folders:
        print("No sample folders found!")
        return
    
    sample_folders.sort()  # Process in order
    
    print(f"Found {len(sample_folders)} sample folders")
    print("Running with VERY HIGH LIMITS for large HTML files!")
    print("Using Qwen2.5-VL model with vision capabilities!")
    
    # Print GPU memory info
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")
    
    for sample_folder in sample_folders:
        sample_path = os.path.join(samples_dir, sample_folder)
        process_sample_folder(sample_path)
    
    print("=== PROCESSING COMPLETE ===")

if __name__ == "__main__":
    main()