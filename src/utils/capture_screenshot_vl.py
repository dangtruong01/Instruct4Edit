import json
import os
import tempfile
from pathlib import Path
from capture_screenshot import capture_screenshot
import time

def prepare_vl_dataset_from_json():
    """Convert existing train.json dataset to VL format with images"""
    
    # Load existing dataset
    with open("data/train.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from train.json")
    
    # Create images directory
    images_dir = Path("data/images")
    images_dir.mkdir(exist_ok=True)
    
    # Check for existing progress
    vl_dataset = []
    processed_indices = set()
    
    if Path("data/vl_train.json").exists():
        try:
            with open("data/vl_train.json", "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                vl_dataset = existing_data
                processed_indices = {item['id'] for item in existing_data}
                print(f"Found existing progress: {len(processed_indices)} samples already processed")
        except:
            print("Could not load existing progress, starting fresh")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for i, item in enumerate(data):
        # Use original sample ID if it exists, otherwise use index
        sample_id = item.get('id', item.get('sample_id', i))
        
        print(f"Processing sample {i+1}/{len(data)} (ID: {sample_id})")
        
        # Check if images already exist on disk
        original_image_path = images_dir / f"original_{sample_id}.png"
        modified_image_path = images_dir / f"modified_{sample_id}.png"
        
        images_exist = original_image_path.exists() and modified_image_path.exists()
        
        # Check if already in our processed dataset
        already_in_dataset = sample_id in processed_indices
        
        if images_exist and already_in_dataset:
            print(f"  ✓ Sample {sample_id} already fully processed, skipping...")
            skipped_count += 1
            continue
        elif images_exist and not already_in_dataset:
            print(f"  ✓ Images exist for sample {sample_id}, adding to dataset...")
            # Add to dataset without regenerating images
            vl_item = {
                "id": sample_id,
                "instruction": item['instruction'],
                "original_html": item['original_html'],
                "modified_html": item['modified_html'],
                "original_image": str(original_image_path),
                "modified_image": str(modified_image_path)
            }
            vl_dataset.append(vl_item)
            success_count += 1
            
            # Save progress every 10 samples
            if (success_count + error_count) % 10 == 0:
                save_progress(vl_dataset)
            continue
        elif not images_exist and already_in_dataset:
            print(f"  ⚠️  Sample {sample_id} in dataset but images missing, regenerating...")
            # Remove from processed set to force regeneration
            processed_indices.discard(sample_id)
            # Remove from vl_dataset
            vl_dataset = [item for item in vl_dataset if item['id'] != sample_id]
        else:
            print(f"  🔄 Sample {sample_id} needs processing...")
        
        # Only process if images don't exist
        if not images_exist:
            # Create temporary HTML files
            original_html_file = None
            modified_html_file = None
            
            try:
                original_html_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
                modified_html_file = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
                
                # Write HTML content to temp files
                original_html_file.write(item['original_html'])
                original_html_file.close()
                
                modified_html_file.write(item['modified_html'])
                modified_html_file.close()
                
                # Capture screenshots with error handling
                print(f"  📸 Capturing original screenshot...")
                success1 = capture_screenshot_safe(
                    Path(original_html_file.name), 
                    original_image_path, 
                    viewport_size=(1280, 720)
                )
                
                if success1:
                    print(f"  📸 Capturing modified screenshot...")
                    success2 = capture_screenshot_safe(
                        Path(modified_html_file.name), 
                        modified_image_path, 
                        viewport_size=(1280, 720)
                    )
                else:
                    success2 = False
                
                if success1 and success2:
                    # Create VL dataset entry
                    vl_item = {
                        "id": sample_id,
                        "instruction": item['instruction'],
                        "original_html": item['original_html'],
                        "modified_html": item['modified_html'],
                        "original_image": str(original_image_path),
                        "modified_image": str(modified_image_path)
                    }
                    
                    vl_dataset.append(vl_item)
                    success_count += 1
                    print(f"  ✓ Generated images for sample {sample_id}")
                else:
                    error_count += 1
                    print(f"  ✗ Failed to generate images for sample {sample_id}")
                    # Clean up partial files
                    if original_image_path.exists():
                        original_image_path.unlink()
                    if modified_image_path.exists():
                        modified_image_path.unlink()
                    
            except KeyboardInterrupt:
                print(f"\n⚠️  Interrupted by user at sample {sample_id}")
                # Clean up partial files
                if original_image_path.exists() and not modified_image_path.exists():
                    original_image_path.unlink()
                break
            except Exception as e:
                error_count += 1
                print(f"  ✗ Error processing sample {sample_id}: {e}")
                # Clean up partial files
                if original_image_path.exists():
                    original_image_path.unlink()
                if modified_image_path.exists():
                    modified_image_path.unlink()
            
            finally:
                # Clean up temp files
                if original_html_file:
                    try:
                        os.unlink(original_html_file.name)
                    except:
                        pass
                if modified_html_file:
                    try:
                        os.unlink(modified_html_file.name)
                    except:
                        pass
            
            # Save progress every 10 samples
            if (success_count + error_count) % 10 == 0:
                save_progress(vl_dataset)
                print(f"  💾 Progress saved: {success_count} success, {error_count} errors, {skipped_count} skipped")
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
    
    # Final save
    save_progress(vl_dataset)
    
    print(f"\n✓ VL dataset preparation complete!")
    print(f"  Total samples processed: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Skipped: {skipped_count}")
    print(f"  Final dataset size: {len(vl_dataset)}")
    
    return len(vl_dataset)

def capture_screenshot_safe(html_path, screenshot_path, viewport_size=(1280, 720), max_retries=2):
    """Safely capture screenshot with retries and timeout handling"""
    for attempt in range(max_retries):
        try:
            capture_screenshot(html_path, screenshot_path, viewport_size)
            # Verify the screenshot was actually created
            if screenshot_path.exists() and screenshot_path.stat().st_size > 0:
                return True
            else:
                raise Exception("Screenshot file not created or empty")
        except Exception as e:
            print(f"    ⚠️  Screenshot attempt {attempt + 1} failed: {str(e)[:100]}...")
            if attempt < max_retries - 1:
                print(f"    🔄 Retrying in 2 seconds...")
                time.sleep(2)
                # Clean up failed screenshot if it exists
                if screenshot_path.exists():
                    screenshot_path.unlink()
            else:
                print(f"    ❌ All attempts failed for {html_path}")
                # Clean up failed screenshot if it exists
                if screenshot_path.exists():
                    screenshot_path.unlink()
                return False
    return False

def save_progress(vl_dataset):
    """Save current progress to file"""
    try:
        # Sort by ID for consistency
        vl_dataset_sorted = sorted(vl_dataset, key=lambda x: x['id'])
        with open("data/vl_train.json", "w", encoding="utf-8") as f:
            json.dump(vl_dataset_sorted, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️  Could not save progress: {e}")

def retry_failed_samples():
    """Retry only the samples that failed previously"""
    print("🔄 Retrying failed samples...")
    
    # Load original data
    with open("data/train.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # Check which samples need retry
    images_dir = Path("data/images")
    failed_samples = []
    
    for item in all_data:
        sample_id = item.get('id', item.get('sample_id', 'unknown'))
        original_image_path = images_dir / f"original_{sample_id}.png"
        modified_image_path = images_dir / f"modified_{sample_id}.png"
        
        if not (original_image_path.exists() and modified_image_path.exists()):
            failed_samples.append(item)
    
    print(f"Found {len(failed_samples)} samples to retry")
    
    if failed_samples:
        # Create a temporary file with only failed samples
        with open("data/temp_retry.json", "w", encoding="utf-8") as f:
            json.dump(failed_samples, f, indent=2, ensure_ascii=False)
        
        # Temporarily replace the original file path in the function
        return prepare_vl_dataset_from_json()
    else:
        print("No failed samples found!")

def resume_from_interruption():
    """Resume processing from where we left off"""
    print("🔄 Resuming interrupted processing...")
    return prepare_vl_dataset_from_json()

if __name__ == "__main__":
    try:
        prepare_vl_dataset_from_json()
    except KeyboardInterrupt:
        print(f"\n⚠️  Processing interrupted by user")
        print("💾 Progress has been saved to data/vl_train.json")
        print("🔄 Run the script again to resume from where you left off")
        print("🔄 Or run retry_failed_samples() to only retry failed cases")