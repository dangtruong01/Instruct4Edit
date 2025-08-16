import os
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def capture_screenshot(html_path: Path, screenshot_path: Path, viewport_size=(1280, 720)):
    """Capture full page screenshot using Playwright"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": viewport_size[0], "height": viewport_size[1]})
        
        # Wait for page to load completely
        page.goto(f"file://{html_path.resolve()}")
        page.wait_for_load_state('networkidle')
        
        # Take full page screenshot
        page.screenshot(path=str(screenshot_path), full_page=True)
        browser.close()

def screenshot_html_file(html_file_path, output_image_path, viewport_size=(1280, 720)):
    """Take a screenshot of an HTML file using Playwright"""
    try:
        html_path = Path(html_file_path)
        screenshot_path = Path(output_image_path)
        
        # Ensure output directory exists
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        
        capture_screenshot(html_path, screenshot_path, viewport_size)
        print(f"Screenshot saved: {output_image_path}")
        return True
    except Exception as e:
        print(f"Error taking screenshot of {html_file_path}: {e}")
        return False

def screenshot_evaluation_samples(overwrite=True):
    """Screenshot all HTML files in evaluation samples"""
    base_dirs = [
        # "gpu/evaluated_samples_2",
        # "evaluated_samples_gemini",
        # "evaluated_samples_base"
        "evaluated_samples_gpt",
        "evaluated_samples_qwen_vl"
    ]
    
    total_processed = 0
    total_skipped = 0
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"Directory {base_dir} does not exist, skipping...")
            continue
            
        print(f"Processing {base_dir}...")
        
        # Get all sample folders
        sample_folders = [f for f in os.listdir(base_dir) 
                         if os.path.isdir(os.path.join(base_dir, f)) and f.startswith('sample_')]
        
        sample_folders.sort()  # Process in order
        
        for sample_folder in sample_folders:
            sample_path = os.path.join(base_dir, sample_folder)
            print(f"  Processing {sample_folder}...")
            
            # Find all HTML files in the sample folder
            html_files = [f for f in os.listdir(sample_path) if f.endswith('.html')]
            
            for html_file in html_files:
                html_path = os.path.join(sample_path, html_file)
                image_name = html_file.replace('.html', '.png')
                image_path = os.path.join(sample_path, image_name)
                
                # Check if image already exists
                if os.path.exists(image_path) and not overwrite:
                    print(f"    Skipping {image_name} (already exists)")
                    total_skipped += 1
                    continue
                
                print(f"    Creating: {image_name}")
                success = screenshot_html_file(html_path, image_path)
                
                if success:
                    total_processed += 1
                else:
                    print(f"    Failed to create screenshot for {html_file}")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
    
    print(f"\n=== SCREENSHOT GENERATION COMPLETE ===")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {total_skipped}")

def screenshot_specific_samples(sample_patterns=None, overwrite=True):
    """Screenshot specific samples based on patterns"""
    base_dirs = [
        # "gpu/evaluated_samples_2",
        # "evaluated_samples_gemini",
        # "evaluated_samples_base"
        "evaluated_samples_gpt",
        "evaluated_samples_qwen_vl"
    ]
    
    if sample_patterns is None:
        sample_patterns = ['sample_']  # Default to all samples
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        print(f"Processing {base_dir}...")
        
        for sample_folder in os.listdir(base_dir):
            sample_path = os.path.join(base_dir, sample_folder)
            
            if not os.path.isdir(sample_path):
                continue
                
            # Check if sample matches any pattern
            if not any(pattern in sample_folder for pattern in sample_patterns):
                continue
            
            print(f"  Processing {sample_folder}...")
            
            for filename in os.listdir(sample_path):
                if filename.endswith('.html'):
                    html_path = os.path.join(sample_path, filename)
                    image_name = filename.replace('.html', '.png')
                    image_path = os.path.join(sample_path, image_name)
                    
                    if os.path.exists(image_path) and not overwrite:
                        print(f"    Skipping {image_name} (already exists)")
                        continue
                    
                    print(f"    Creating: {image_name}")
                    screenshot_html_file(html_path, image_path)

def check_missing_screenshots():
    """Check which HTML files are missing corresponding screenshots"""
    base_dirs = [
        # "gpu/evaluated_samples_2",
        # "evaluated_samples_gemini",
        # "evaluated_samples_base",
        "evaluated_samples_gpt",
        "evaluated_samples_qwen_vl"
    ]
    
    missing_count = 0
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        print(f"Checking {base_dir}...")
        
        for sample_folder in os.listdir(base_dir):
            sample_path = os.path.join(base_dir, sample_folder)
            
            if not os.path.isdir(sample_path):
                continue
                
            missing_in_sample = []
            
            for filename in os.listdir(sample_path):
                if filename.endswith('.html'):
                    image_name = filename.replace('.html', '.png')
                    image_path = os.path.join(sample_path, image_name)
                    
                    if not os.path.exists(image_path):
                        missing_in_sample.append(filename)
                        missing_count += 1
            
            if missing_in_sample:
                print(f"  {sample_folder}: Missing {len(missing_in_sample)} screenshots")
                for missing in missing_in_sample:
                    print(f"    - {missing}")
    
    print(f"\nTotal missing screenshots: {missing_count}")
    return missing_count > 0

if __name__ == "__main__":
    # First check what's missing
    print("=== CHECKING FOR MISSING SCREENSHOTS ===")
    has_missing = check_missing_screenshots()
    
    if has_missing:
        print("\n=== GENERATING MISSING SCREENSHOTS ===")
        # Generate all screenshots, overwriting existing ones
        screenshot_evaluation_samples(overwrite=True)
    else:
        print("\n✓ All screenshots already exist!")
        
    # Optional: Generate screenshots for specific samples only
    # screenshot_specific_samples(['sample_1', 'sample_2'], overwrite=True)