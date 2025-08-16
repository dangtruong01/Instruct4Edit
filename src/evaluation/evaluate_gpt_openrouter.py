import os
import json
import time
import requests
from datetime import datetime

# Configuration for OpenRouter
api_key = os.getenv("OPENAI_API_KEY")  # Your OpenRouter API key
samples_dir = r"D:\DATA1\SMU\yutong\DesignEdit-model-training\evaluated_samples_gpt"
MODEL_NAME = "openai/gpt-4o-mini"

def verify_api_key():
    """Verify that the OpenRouter API key is valid"""
    if not api_key:
        print("❌ No API key found in environment variable OPENAI_API_KEY")
        return False
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/test/designedit",
        "X-Title": "DesignEdit Test"
    }
    
    data = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Test"}],
        "max_tokens": 5
    }
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ OpenRouter API key is valid!")
            return True
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def generate_output(instruction, original_html):
    """Generate GPT output for given instruction and HTML using direct HTTP requests"""
    prompt = (
        "You are a frontend developer. Given the HTML code and a design instruction, "
        "apply the change and return the full modified HTML file. "
        "Do NOT skip any parts of the code — output the complete modified HTML with only the necessary edits.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Original HTML:\n{original_html}\n\n"
        "Modified HTML:"
    )

    print(f"    Input length: {len(prompt)} characters")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/test/designedit",
        "X-Title": "DesignEdit Evaluation"
    }

    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a skilled frontend developer who modifies HTML code based on design instructions."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 16384,
        "temperature": 0
    }

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=120
        )
        
        print(f"    HTTP Status: {response.status_code}")
        
        if response.status_code == 429:  # Rate limit
            print("    Rate limit exceeded. Waiting 60 seconds...")
            time.sleep(60)
            return generate_output(instruction, original_html)  # Retry
        
        if response.status_code != 200:
            error_msg = f"HTTP {response.status_code} - {response.text}"
            print(f"    Error: {error_msg}")
            return f"ERROR: {error_msg}"
        
        result = response.json()
        
        if "choices" not in result or not result["choices"]:
            print(f"    Invalid response structure: {result}")
            return f"ERROR: Invalid response structure"
        
        output_text = result["choices"][0]["message"]["content"]
        
        # Print usage if available
        if "usage" in result:
            usage = result["usage"]
            print(f"    Tokens used - Input: {usage.get('prompt_tokens', 'N/A')}, Output: {usage.get('completion_tokens', 'N/A')}, Total: {usage.get('total_tokens', 'N/A')}")
        
        return output_text
        
    except requests.exceptions.Timeout:
        print("    Request timeout")
        return "ERROR: Request timeout"
    except requests.exceptions.RequestException as e:
        print(f"    Request failed: {e}")
        return f"ERROR: Request failed - {str(e)}"
    except Exception as e:
        print(f"    Unexpected error: {e}")
        return f"ERROR: {str(e)}"

def check_instruction_status(sample_path, instruction_index):
    """Check if an instruction was processed successfully or had an error"""
    # Check for output file
    output_file = f"output_gpt_{instruction_index}.html"
    output_path = os.path.join(sample_path, output_file)
    
    # Check for error file
    error_file = f"error_gpt_{instruction_index}.txt"
    error_path = os.path.join(sample_path, error_file)
    
    if os.path.exists(output_path):
        # Check if the file is reasonably sized (not empty or truncated)
        file_size = os.path.getsize(output_path)
        if file_size > 100:  # At least 100 bytes
            return "completed"
        else:
            return "incomplete"
    elif os.path.exists(error_path):
        return "error"
    else:
        return "missing"

def process_sample_folder(sample_path, retry_only=False):
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
        # Extract instruction number from filename
        inst_num = inst_file.replace('instruction_', '').replace('.txt', '').replace('.json', '')
        
        # Check status if in retry mode
        if retry_only:
            status = check_instruction_status(sample_path, inst_num)
            if status == "completed":
                print(f"  Instruction {inst_num}: Already completed, skipping")
                continue
            elif status == "error":
                print(f"  Instruction {inst_num}: Found error, retrying...")
            elif status == "incomplete":
                print(f"  Instruction {inst_num}: Incomplete output, retrying...")
            else:
                print(f"  Instruction {inst_num}: Missing output, processing...")
        
        inst_path = os.path.join(sample_path, inst_file)
        
        # Read instruction
        with open(inst_path, 'r', encoding='utf-8') as f:
            if inst_file.endswith('.json'):
                instruction_data = json.load(f)
                instruction = instruction_data.get('instruction', instruction_data.get('text', str(instruction_data)))
            else:
                instruction = f.read().strip()
        
        print(f"  Processing instruction {inst_num}: {instruction[:50]}...")
        print(f"    Original HTML length: {len(original_html)} characters")
        
        # Generate output with timing
        start_time = time.time()
        output = generate_output(instruction, original_html)
        end_time = time.time()
        
        print(f"    Generation completed in {end_time - start_time:.2f} seconds")
        
        # Handle errors
        if output.startswith("ERROR:"):
            print(f"    ERROR during generation: {output}")
            error_file = f"error_gpt_{inst_num}.txt"
            error_path = os.path.join(sample_path, error_file)
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Error: {output}\nInstruction: {instruction}\nTimestamp: {datetime.now()}\n")
            continue
        
        # Save output
        output_file = f"output_gpt_{inst_num}.html"
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
        debug_file = f"debug_output_gpt_{inst_num}.txt"
        debug_path = os.path.join(sample_path, debug_file)
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Instruction: {instruction}\n")
            f.write("="*50 + "\n")
            f.write(output)
        print(f"  📄 Debug output saved to {debug_file}")
        
        # Rate limiting - be respectful to the API
        time.sleep(1)  # 1 second between requests

def scan_for_errors():
    """Scan all sample folders for errors and incomplete outputs"""
    if not os.path.exists(samples_dir):
        print(f"Samples directory {samples_dir} not found!")
        return
    
    sample_folders = [f for f in os.listdir(samples_dir) 
                     if os.path.isdir(os.path.join(samples_dir, f)) and f.startswith('sample_')]
    sample_folders.sort()
    
    print("=== SCANNING FOR ERRORS AND INCOMPLETE OUTPUTS ===")
    
    total_errors = 0
    total_incomplete = 0
    total_missing = 0
    
    for sample_folder in sample_folders:
        sample_path = os.path.join(samples_dir, sample_folder)
        
        # Find instruction files
        instruction_files = []
        for file in os.listdir(sample_path):
            if file.startswith('instruction_') and (file.endswith('.txt') or file.endswith('.json')):
                instruction_files.append(file)
        
        if not instruction_files:
            continue
        
        print(f"\n{sample_folder}:")
        folder_issues = False
        
        for inst_file in instruction_files:
            inst_num = inst_file.replace('instruction_', '').replace('.txt', '').replace('.json', '')
            status = check_instruction_status(sample_path, inst_num)
            
            if status != "completed":
                folder_issues = True
                if status == "error":
                    print(f"  ❌ Instruction {inst_num}: ERROR")
                    total_errors += 1
                elif status == "incomplete":
                    print(f"  ⚠️  Instruction {inst_num}: INCOMPLETE")
                    total_incomplete += 1
                elif status == "missing":
                    print(f"  ❓ Instruction {inst_num}: MISSING")
                    total_missing += 1
        
        if not folder_issues:
            print(f"  ✅ All instructions completed successfully")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total errors: {total_errors}")
    print(f"Total incomplete: {total_incomplete}")
    print(f"Total missing: {total_missing}")
    total_issues = total_errors + total_incomplete + total_missing
    if total_issues > 0:
        print(f"Total issues: {total_issues}")
        print("\nTo retry failed instructions, run: python evaluate_gpt_openrouter.py --retry")
    else:
        print("✅ No issues found!")

def main():
    """Main function to process all sample folders"""
    import sys
    
    # Check for retry flag
    retry_only = "--retry" in sys.argv
    scan_only = "--scan" in sys.argv
    
    # Check API key and verify it works
    if not scan_only:
        print("Verifying OpenRouter API key...")
        if not verify_api_key():
            print("Please check your OpenRouter API key and try again.")
            return
    
    if scan_only:
        scan_for_errors()
        return
    
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
    
    if retry_only:
        print("=== RETRYING FAILED INSTRUCTIONS ===")
    else:
        print("=== STARTING GPT EVALUATION (via OpenRouter) ===")
    
    print(f"Found {len(sample_folders)} sample folders")
    print(f"Using model: {MODEL_NAME}")
    print(f"Starting evaluation at: {datetime.now()}")
    
    if retry_only:
        print("Retry mode: Only processing failed/incomplete instructions")
    
    for sample_folder in sample_folders:
        sample_path = os.path.join(samples_dir, sample_folder)
        process_sample_folder(sample_path, retry_only=retry_only)
    
    print("=== GPT EVALUATION COMPLETE ===")
    print(f"Completed at: {datetime.now()}")
    
    # Run scan after completion
    if not retry_only:
        print("\n" + "="*50)
        scan_for_errors()

if __name__ == "__main__":
    main()