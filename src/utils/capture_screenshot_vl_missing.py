import json
from pathlib import Path

def find_uncaptured_samples():
    """
    Find all samples from train.json that don't have images captured yet
    and save them to a new JSON file
    """
    
    # Load original dataset
    try:
        with open("data/train.json", "r", encoding="utf-8") as f:
            all_data = json.load(f)
        print(f"✓ Loaded {len(all_data)} samples from train.json")
    except FileNotFoundError:
        print("❌ Error: data/train.json not found!")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing train.json: {e}")
        return
    
    # Check images directory
    images_dir = Path("data/images")
    if not images_dir.exists():
        print(f"📁 Images directory doesn't exist: {images_dir}")
        print("   All samples need image capture")
        uncaptured_samples = all_data
    else:
        print(f"📁 Checking for existing images in: {images_dir}")
        
        # Find samples without images
        uncaptured_samples = []
        
        for i, item in enumerate(all_data):
            # Use same ID logic as the original script
            sample_id = item.get('id', item.get('sample_id', i))
            
            # Check if both images exist
            original_image_path = images_dir / f"original_{sample_id}.png"
            modified_image_path = images_dir / f"modified_{sample_id}.png"
            
            images_exist = (
                original_image_path.exists() and 
                modified_image_path.exists() and
                original_image_path.stat().st_size > 0 and
                modified_image_path.stat().st_size > 0
            )
            
            if not images_exist:
                uncaptured_samples.append(item)
                print(f"  🔍 Sample {sample_id}: Missing images")
            else:
                print(f"  ✓ Sample {sample_id}: Images exist")
    
    # Save uncaptured samples to new file
    output_file = "data/uncaptured_samples.json"
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(uncaptured_samples, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Summary:")
        print(f"  Total samples in train.json: {len(all_data)}")
        print(f"  Samples with images: {len(all_data) - len(uncaptured_samples)}")
        print(f"  Samples without images: {len(uncaptured_samples)}")
        print(f"  Coverage: {((len(all_data) - len(uncaptured_samples)) / len(all_data) * 100):.1f}%")
        print(f"\n💾 Uncaptured samples saved to: {output_file}")
        
        if len(uncaptured_samples) == 0:
            print("🎉 All samples already have images captured!")
        else:
            print(f"🔄 {len(uncaptured_samples)} samples still need image capture")
        
        return uncaptured_samples
        
    except Exception as e:
        print(f"❌ Error saving uncaptured samples: {e}")
        return None

def get_detailed_status():
    """
    Get detailed status of each sample's image capture state
    """
    
    # Load original dataset
    try:
        with open("data/train.json", "r", encoding="utf-8") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print("❌ Error: data/train.json not found!")
        return
    
    images_dir = Path("data/images")
    
    status_report = {
        "complete": [],      # Both images exist
        "partial": [],       # Only one image exists
        "missing": [],       # No images exist
        "corrupted": []      # Images exist but are empty/corrupted
    }
    
    for i, item in enumerate(all_data):
        sample_id = item.get('id', item.get('sample_id', i))
        
        original_image_path = images_dir / f"original_{sample_id}.png"
        modified_image_path = images_dir / f"modified_{sample_id}.png"
        
        original_exists = original_image_path.exists()
        modified_exists = modified_image_path.exists()
        
        # Check file sizes if they exist
        original_valid = original_exists and original_image_path.stat().st_size > 0
        modified_valid = modified_exists and modified_image_path.stat().st_size > 0
        
        sample_info = {
            "id": sample_id,
            "instruction": item['instruction'][:100] + "..." if len(item['instruction']) > 100 else item['instruction'],
            "original_exists": original_exists,
            "modified_exists": modified_exists,
            "original_size": original_image_path.stat().st_size if original_exists else 0,
            "modified_size": modified_image_path.stat().st_size if modified_exists else 0
        }
        
        if original_valid and modified_valid:
            status_report["complete"].append(sample_info)
        elif (original_exists and not original_valid) or (modified_exists and not modified_valid):
            status_report["corrupted"].append(sample_info)
        elif original_exists or modified_exists:
            status_report["partial"].append(sample_info)
        else:
            status_report["missing"].append(sample_info)
    
    # Save detailed report
    with open("data/image_capture_status.json", "w", encoding="utf-8") as f:
        json.dump(status_report, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n📊 Detailed Status Report:")
    print(f"  ✅ Complete (both images): {len(status_report['complete'])}")
    print(f"  ⚠️  Partial (one image): {len(status_report['partial'])}")
    print(f"  ❌ Missing (no images): {len(status_report['missing'])}")
    print(f"  💥 Corrupted (empty files): {len(status_report['corrupted'])}")
    print(f"\n📄 Detailed report saved to: data/image_capture_status.json")
    
    return status_report

def create_retry_dataset():
    """
    Create a dataset specifically for retrying failed/missing captures
    Includes both missing and corrupted samples
    """
    
    status = get_detailed_status()
    if not status:
        return
    
    # Combine samples that need processing
    retry_samples = []
    
    # Load original data to get full sample info
    with open("data/train.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)
    
    # Create lookup by ID
    data_lookup = {}
    for i, item in enumerate(all_data):
        sample_id = item.get('id', item.get('sample_id', i))
        data_lookup[sample_id] = item
    
    # Add missing samples
    for sample_info in status["missing"]:
        if sample_info["id"] in data_lookup:
            retry_samples.append(data_lookup[sample_info["id"]])
    
    # Add partial samples
    for sample_info in status["partial"]:
        if sample_info["id"] in data_lookup:
            retry_samples.append(data_lookup[sample_info["id"]])
    
    # Add corrupted samples
    for sample_info in status["corrupted"]:
        if sample_info["id"] in data_lookup:
            retry_samples.append(data_lookup[sample_info["id"]])
    
    # Save retry dataset
    output_file = "data/retry_capture.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(retry_samples, f, indent=2, ensure_ascii=False)
    
    print(f"\n🔄 Retry dataset created: {output_file}")
    print(f"   Contains {len(retry_samples)} samples that need (re)processing")
    
    return retry_samples

if __name__ == "__main__":
    print("🔍 Finding uncaptured samples...")
    uncaptured = find_uncaptured_samples()
    
    print("\n" + "="*50)
    print("📊 Getting detailed status...")
    status = get_detailed_status()
    
    print("\n" + "="*50)
    print("🔄 Creating retry dataset...")
    retry_data = create_retry_dataset()
    
    if uncaptured is not None:
        print(f"\n✅ Analysis complete!")
        print(f"📁 Files created:")
        print(f"   • data/uncaptured_samples.json - {len(uncaptured)} samples without images")
        print(f"   • data/image_capture_status.json - Detailed status report")
        print(f"   • data/retry_capture.json - {len(retry_data)} samples for retry")