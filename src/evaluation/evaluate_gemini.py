import google.generativeai as genai
import os
import json
import time
from datetime import datetime

# Configuration
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Set your Google API key as environment variable
samples_dir = "D:\DATA1\SMU\yutong\DesignEdit-model-training\evaluated_samples_gemini"
MODEL_NAME = "gemini-2.5-pro"  # You can also use "gemini-1.5-flash" for faster inference

# Initialize the model
model = genai.GenerativeModel(MODEL_NAME)

def generate_output(instruction, original_html):
    """Generate Gemini output for given instruction and HTML"""
    prompt = (
        "You are a frontend developer. Given the HTML code and a design instruction, "
        "apply the change and return the full modified HTML file. "
        "Do NOT skip any parts of the code — output the complete modified HTML with only the necessary edits.\n\n"
        f"Instruction:\n{instruction}\n\n"
        f"Original HTML:\n{original_html}\n\n"
        "Modified HTML:"
    )

    print(f"    Input length: {len(prompt)} characters")

    try:
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=8192,  # Maximum tokens for response
            temperature=0,           # Deterministic output
            top_p=1,
            top_k=1
        )
        
        # Make API call to Gemini
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                }
            ]
        )
        
        # Check if response was blocked
        if response.prompt_feedback.block_reason:
            return f"ERROR: Content blocked - {response.prompt_feedback.block_reason}"
        
        if not response.parts:
            return "ERROR: No response generated"
        
        output_text = response.text
        
        # Print token usage info if available
        try:
            usage = response.usage_metadata
            print(f"    Tokens used - Input: {usage.prompt_token_count}, Output: {usage.candidates_token_count}, Total: {usage.total_token_count}")
        except:
            print("    Token usage info not available")
        
        return output_text
        
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            print(f"    Rate limit/quota exceeded: {e}")
            print("    Waiting 60 seconds before retry...")
            time.sleep(60)
            return generate_output(instruction, original_html)  # Retry
        else:
            print(f"    API error: {e}")
            return f"ERROR: {str(e)}"

def check_instruction_status(sample_path, instruction_index):
    """Check if an instruction was processed successfully or had an error"""
    output_file = f"gemini_output_{instruction_index}.html"
    error_file = f"error_output_{instruction_index}.txt"
    
    output_path = os.path.join(sample_path, output_file)
    error_path = os.path.join(sample_path, error_file)
    
    # Check if successful output exists and is valid
    if os.path.exists(output_path):
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Check if it's a valid HTML file (basic check)
                if content and ('<!DOCTYPE' in content or '<html' in content) and len(content) > 100:
                    return "success"
                else:
                    return "incomplete"
        except:
            return "error"
    
    # Check if error file exists
    if os.path.exists(error_path):
        return "error"
    
    return "not_processed"

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
        inst_path = os.path.join(sample_path, inst_file)
        
        # Check current status
        status = check_instruction_status(sample_path, i)
        
        if retry_only and status == "success":
            print(f"  ✓ Instruction {i} already completed successfully - skipping")
            continue
        elif retry_only and status == "not_processed":
            print(f"  → Instruction {i} not processed yet - skipping (retry mode)")
            continue
        elif status == "success" and not retry_only:
            print(f"  ✓ Instruction {i} already completed successfully - skipping")
            continue
        elif status == "error":
            print(f"  🔄 Instruction {i} had errors - retrying...")
            # Remove old error file
            error_file = f"error_output_{i}.txt"
            error_path = os.path.join(sample_path, error_file)
            if os.path.exists(error_path):
                os.remove(error_path)
        elif status == "incomplete":
            print(f"  🔄 Instruction {i} was incomplete - retrying...")
        else:
            print(f"  → Processing instruction {i} (new)...")
        
        # Read instruction
        with open(inst_path, 'r', encoding='utf-8') as f:
            if inst_file.endswith('.json'):
                instruction_data = json.load(f)
                instruction = instruction_data.get('instruction', instruction_data.get('text', str(instruction_data)))
            else:
                instruction = f.read().strip()
        
        print(f"    Instruction: {instruction[:50]}...")
        print(f"    Original HTML length: {len(original_html)} characters")
        
        # Generate output with timing
        start_time = time.time()
        output = generate_output(instruction, original_html)
        end_time = time.time()
        
        print(f"    Generation completed in {end_time - start_time:.2f} seconds")
        
        # Handle errors
        if output.startswith("ERROR:"):
            print(f"    ERROR during generation: {output}")
            error_file = f"error_output_{i}.txt"
            error_path = os.path.join(sample_path, error_file)
            with open(error_path, 'w', encoding='utf-8') as f:
                f.write(f"Error: {output}\nInstruction: {instruction}\nTimestamp: {datetime.now()}\n")
            continue
        
        # Save output
        output_file = f"gemini_output_{i}.html"
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
        debug_file = f"gemini_debug_output_{i}.txt"
        debug_path = os.path.join(sample_path, debug_file)
        with open(debug_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.write(f"Instruction: {instruction}\n")
            f.write("="*50 + "\n")
            f.write(output)
        print(f"  📄 Debug output saved to {debug_file}")
        
        # Rate limiting - be respectful to the API
        time.sleep(2)  # 2 seconds between requests for Gemini

def scan_for_errors():
    """Scan all sample folders for errors and incomplete outputs"""
    print("=== SCANNING FOR ERRORS AND INCOMPLETE OUTPUTS ===")
    
    if not os.path.exists(samples_dir):
        print(f"Samples directory {samples_dir} not found!")
        return
    
    sample_folders = [f for f in os.listdir(samples_dir) 
                     if os.path.isdir(os.path.join(samples_dir, f)) and f.startswith('sample_')]
    
    total_errors = 0
    total_incomplete = 0
    total_success = 0
    
    error_summary = []
    
    for sample_folder in sorted(sample_folders):
        sample_path = os.path.join(samples_dir, sample_folder)
        
        # Count instruction files
        instruction_files = [f for f in os.listdir(sample_path) 
                           if f.startswith('instruction_') and (f.endswith('.txt') or f.endswith('.json'))]
        
        sample_errors = 0
        sample_incomplete = 0
        sample_success = 0
        
        for i in range(1, len(instruction_files) + 1):
            status = check_instruction_status(sample_path, i)
            
            if status == "error":
                sample_errors += 1
                total_errors += 1
            elif status == "incomplete":
                sample_incomplete += 1
                total_incomplete += 1
            elif status == "success":
                sample_success += 1
                total_success += 1
        
        if sample_errors > 0 or sample_incomplete > 0:
            error_summary.append(f"{sample_folder}: {sample_errors} errors, {sample_incomplete} incomplete, {sample_success} success")
        
        print(f"{sample_folder}: {sample_success} success, {sample_errors} errors, {sample_incomplete} incomplete")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total successful: {total_success}")
    print(f"Total errors: {total_errors}")
    print(f"Total incomplete: {total_incomplete}")
    
    if error_summary:
        print(f"\nFolders with issues:")
        for line in error_summary:
            print(f"  {line}")
    
    return total_errors + total_incomplete > 0

def main():
    """Main function to process all sample folders"""
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Google API key not found!")
        print("Please set your API key: export GOOGLE_API_KEY='your-api-key-here'")
        return
    
    if not os.path.exists(samples_dir):
        print(f"Samples directory {samples_dir} not found!")
        return
    
    # First scan for errors
    has_issues = scan_for_errors()
    
    if not has_issues:
        print("\n✓ No errors or incomplete outputs found!")
        return
    
    # Ask user if they want to retry
    print(f"\nUsing Google model: {MODEL_NAME}")
    retry_choice = input("\nWould you like to retry failed/incomplete instructions? (y/n): ").lower().strip()
    
    if retry_choice != 'y':
        print("Exiting without retrying.")
        return
    
    # Get all sample folders
    sample_folders = [f for f in os.listdir(samples_dir) 
                     if os.path.isdir(os.path.join(samples_dir, f)) and f.startswith('sample_')]
    
    sample_folders.sort()  # Process in order
    
    print(f"\nStarting retry evaluation at: {datetime.now()}")
    
    for sample_folder in sample_folders:
        sample_path = os.path.join(samples_dir, sample_folder)
        process_sample_folder(sample_path, retry_only=True)
    
    print("\n=== RETRY EVALUATION COMPLETE ===")
    print(f"Completed at: {datetime.now()}")
    
    # Final scan
    print("\n=== FINAL STATUS ===")
    scan_for_errors()

if __name__ == "__main__":
    main()