import torch
import os
import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Paths - update to match your actual paths
base_model_path = "/root/Qwen2.5-7B-Instruct-1M"  # or "/root/Qwen2.5-7B-Instruct-1M" if on Linux
lora_weights_path = "./models/qwen2.5-7b-instruct-1m-finetuned-design-edit"
samples_dir = "/root/evaluated_samples"

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# Load base model on GPU
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # Use float16 for GPU
    trust_remote_code=True,
    device_map="auto"  # Auto distribute across available GPUs
)

# Load LoRA model
model = PeftModel.from_pretrained(base_model, lora_weights_path)
model.eval()

def generate_output(instruction, original_html):
    """Generate model output for given instruction and HTML - VERY HIGH LIMITS"""
    prompt = (
        "You are a frontend developer. Given the HTML code and a design instruction, "
        "apply the change and return the full modified HTML file. "
        "Do NOT skip any parts of the code — output the complete modified HTML with only the necessary edits.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Original HTML:\n{original_html}\n\n"
        "Modified HTML:"
    )

    # Tokenize WITHOUT truncation - let it handle full HTML
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"    Input token count: {inputs['input_ids'].shape[1]}")

    # Generate with VERY HIGH LIMITS for 20k+ character HTML files
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=16384,       # Very high limit for large HTML files (16k tokens)
            do_sample=False,            # Greedy for consistency
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=False,       # Don't stop early
            use_cache=True,             # For efficiency
            repetition_penalty=1.02,    # Slight penalty to avoid loops
            # Remove max_length=None to avoid the error
        )
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return output_text

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
        
        # Generate output with timing
        start_time = time.time()
        try:
            output = generate_output(instruction, original_html)
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
        output_file = f"output_{i}.html"
        output_path = os.path.join(sample_path, output_file)
        
        # Extract just the modified HTML from the model output
        if "Modified HTML:" in output:
            modified_html = output.split("Modified HTML:")[-1].strip()
        else:
            # Fallback: try to find HTML content
            if '<!DOCTYPE' in output:
                start_pos = output.find('<!DOCTYPE')
                modified_html = output[start_pos:]
            elif '<html' in output:
                start_pos = output.find('<html')
                modified_html = output[start_pos:]
            else:
                modified_html = output
        
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
        debug_file = f"debug_output_{i}.txt"
        debug_path = os.path.join(sample_path, debug_file)
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(output)
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