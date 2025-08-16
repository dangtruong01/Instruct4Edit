import json
import shutil
from pathlib import Path

def create_original_images_only():
    """Create a new image folder containing only original images from the VL dataset"""
    
    # Load the VL dataset
    try:
        with open("data/vl_train_unfiltered.json", "r", encoding="utf-8") as f:
            vl_data = json.load(f)
        print(f"Loaded {len(vl_data)} samples from vl_train_unfiltered.json")
    except FileNotFoundError:
        print("Error: data/vl_train_unfiltered.json not found!")
        return
    
    # Create new directory for original images only
    original_images_dir = Path("data/images_original_only")
    original_images_dir.mkdir(exist_ok=True)
    print(f"Created directory: {original_images_dir}")
    
    # Copy only original images
    copied_count = 0
    missing_count = 0
    error_count = 0
    
    for i, item in enumerate(vl_data):
        try:
            # Get original image path - handle both formats
            if 'original_image' in item:
                original_image_path_str = item['original_image']
                # Convert to local path format
                if original_image_path_str.startswith('/root/DesignEdit/'):
                    # Convert from server path to local path
                    local_path = original_image_path_str.replace('/root/DesignEdit/', '')
                else:
                    local_path = original_image_path_str
                
                original_image_path = Path(local_path)
            else:
                print(f"No original_image field in item {i}")
                continue
            
            # Check if original image exists
            if original_image_path.exists():
                # Create destination path
                dest_path = original_images_dir / original_image_path.name
                
                # Copy the file
                shutil.copy2(original_image_path, dest_path)
                copied_count += 1
                
                if (copied_count + missing_count) % 100 == 0:
                    print(f"Processed {copied_count + missing_count}/{len(vl_data)} files...")
                    
            else:
                missing_count += 1
                print(f"Missing: {original_image_path}")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing item {i}: {e}")
    
    print(f"\n✓ Original images copying complete!")
    print(f"  Successfully copied: {copied_count}")
    print(f"  Missing files: {missing_count}")
    print(f"  Errors: {error_count}")
    print(f"  New folder: {original_images_dir}")
    
    # Calculate size savings
    if Path("data/images").exists():
        original_size = sum(f.stat().st_size for f in Path("data/images").rglob('*') if f.is_file())
        new_size = sum(f.stat().st_size for f in original_images_dir.rglob('*') if f.is_file())
        
        print(f"\nStorage comparison:")
        print(f"  Original images folder: {original_size / (1024**3):.2f} GB")
        print(f"  New original-only folder: {new_size / (1024**3):.2f} GB")
        print(f"  Space saved: {(original_size - new_size) / (1024**3):.2f} GB")
    
    return copied_count

def update_dataset_paths():
    """Update the VL dataset to point to the new original-only images folder and remove modified image paths"""
    
    # Load the VL dataset
    with open("data/vl_train_unfiltered.json", "r", encoding="utf-8") as f:
        vl_data = json.load(f)
    
    # Update paths to point to new folder
    for item in vl_data:
        if 'original_image' in item:
            original_path = Path(item['original_image'])
            # Update to new folder path with correct server format
            item['original_image'] = f"/root/DesignEdit/data/images_original_only/{original_path.name}"
        
        # Remove modified_image field since we don't need it
        if 'modified_image' in item:
            del item['modified_image']
    
    # Save updated dataset
    with open("data/vl_train_unfiltered_original_only.json", "w", encoding="utf-8") as f:
        json.dump(vl_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created updated dataset: data/vl_train_unfiltered_original_only.json")
    print(f"  - Updated image paths to images_original_only folder")
    print(f"  - Removed modified_image fields")
    
    return len(vl_data)

def create_tar_for_transfer():
    """Create a tar file for easier transfer of original images"""
    import tarfile
    
    tar_path = Path("data/images_original_only.tar.gz")
    images_dir = Path("data/images_original_only")
    
    if not images_dir.exists():
        print("Error: images_original_only folder doesn't exist. Run create_original_images_only() first.")
        return
    
    print("Creating tar.gz file for transfer...")
    
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(images_dir, arcname="images_original_only")
    
    tar_size = tar_path.stat().st_size / (1024**3)
    print(f"✓ Created {tar_path} ({tar_size:.2f} GB)")
    print(f"  Transfer command: scp -P 41114 {tar_path} root@180.44.78.106:~/DesignEdit/data/")
    print(f"  Extract command: tar -xzf images_original_only.tar.gz")
    
    return str(tar_path)

def check_unfiltered_status():
    """Check status of unfiltered dataset processing"""
    try:
        with open("data/vl_train_unfiltered.json", "r", encoding="utf-8") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: data/vl_train_unfiltered.json not found!")
        return
    
    images_dir = Path("data/images")
    
    complete_count = 0
    partial_count = 0
    missing_count = 0
    
    for item in all_data:
        if 'original_image' in item:
            original_image_path_str = item['original_image']
            # Convert server path to local path for checking
            if original_image_path_str.startswith('/root/DesignEdit/'):
                local_path = original_image_path_str.replace('/root/DesignEdit/', '')
            else:
                local_path = original_image_path_str
            
            original_image_path = Path(local_path)
            original_exists = original_image_path.exists() and original_image_path.stat().st_size > 0
            
            if original_exists:
                complete_count += 1
            else:
                missing_count += 1
        else:
            missing_count += 1
    
    print(f"\n📊 Unfiltered Dataset Status:")
    print(f"  Total samples: {len(all_data)}")
    print(f"  ✅ Have original images: {complete_count}")
    print(f"  ❌ Missing original images: {missing_count}")
    print(f"  Progress: {(complete_count / len(all_data) * 100):.1f}%")
    
    return {
        "total": len(all_data),
        "complete": complete_count,
        "missing": missing_count
    }

def main():
    print("Creating original images only folder for unfiltered dataset...\n")
    
    # Step 1: Check current status
    print("Checking current dataset status...")
    check_unfiltered_status()
    print(f"\n" + "="*50)
    
    # Step 2: Copy original images only
    print("Copying original images...")
    copied_count = create_original_images_only()
    
    if copied_count > 0:
        print(f"\n" + "="*50)
        
        # Step 3: Update dataset paths
        print("Updating dataset paths...")
        update_dataset_paths()
        
        print(f"\n" + "="*50)
        
        # Step 4: Create tar file for transfer
        print("Creating tar file for transfer...")
        tar_path = create_tar_for_transfer()
        
        print(f"\n" + "="*50)
        print("Summary:")
        print(f"✓ New folder: data/images_original_only/")
        print(f"✓ Updated dataset: data/vl_train_unfiltered_original_only.json")
        print(f"✓ Transfer file: {tar_path}")
        print(f"\nTo use the new dataset in train_vl.py, change the input file to:")
        print(f'    with open("data/vl_train_unfiltered_original_only.json", "r", encoding="utf-8") as f:')
    else:
        print("No images were copied. Please check the dataset and image paths.")

if __name__ == "__main__":
    main()