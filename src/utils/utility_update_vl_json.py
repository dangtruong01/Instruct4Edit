import json
import os

def update_image_paths(json_file_path, output_file_path=None):
    """
    Update image paths in vl_train.json from Windows format to Unix format
    
    Args:
        json_file_path: Path to the vl_train.json file
        output_file_path: Optional output path. If None, overwrites the original file
    """
    
    # Load the JSON data
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {json_file_path}")
    
    # Counter for tracking changes
    updated_count = 0
    
    # Update image paths
    for item in data:
        # Update original_image path if it exists
        if 'original_image' in item:
            old_path = item['original_image']
            # Convert Windows backslashes to forward slashes and update prefix
            new_path = old_path.replace('\\', '/').replace('data/images/', '/root/DesignEdit/data/images/')
            item['original_image'] = new_path
            print(f"Updated original_image: {old_path} -> {new_path}")
            updated_count += 1
        
        # Update modified_image path if it exists
        if 'modified_image' in item:
            old_path = item['modified_image']
            # Convert Windows backslashes to forward slashes and update prefix
            new_path = old_path.replace('\\', '/').replace('data/images/', '/root/DesignEdit/data/images/')
            item['modified_image'] = new_path
            print(f"Updated modified_image: {old_path} -> {new_path}")
            updated_count += 1
    
    # Determine output file path
    if output_file_path is None:
        output_file_path = json_file_path
    
    # Save the updated data
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Successfully updated {updated_count} image paths")
    print(f"📄 Updated file saved to: {output_file_path}")
    
    return data

def main():
    """Main function to run the path update script"""
    
    # Input and output file paths
    input_file = "data/vl_train.json"
    output_file = "data/vl_train_updated.json"  # Change to None to overwrite original
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: Input file {input_file} not found!")
        return
    
    try:
        # Update the image paths
        updated_data = update_image_paths(input_file, output_file)
        
        # Show a sample of the changes
        print(f"\n📋 Sample of updated entries:")
        for i, item in enumerate(updated_data[:3]):  # Show first 3 entries
            print(f"  Sample {i+1}:")
            if 'original_image' in item:
                print(f"    Original image: {item['original_image']}")
            if 'modified_image' in item:
                print(f"    Modified image: {item['modified_image']}")
            print()
        
    except Exception as e:
        print(f"❌ Error updating image paths: {e}")

if __name__ == "__main__":
    main()