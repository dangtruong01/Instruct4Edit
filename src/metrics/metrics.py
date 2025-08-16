#!/usr/bin/env python3
"""
DesignEdit Evaluation Script - SSIM + CLIP Visual Metrics
Evaluates HTML design modifications using two complementary visual metrics:
1. SSIM - Structural similarity (layout preservation)
2. CLIP - Semantic visual similarity (content understanding)

Usage:
    python evaluate_designedit.py

Requirements:
    pip install torch torchvision clip-by-openai pillow scikit-image opencv-python
"""

import torch
import clip
from PIL import Image
import numpy as np
from skimage import metrics
import cv2
import os
import json
from datetime import datetime
import sys
import traceback

class DesignEditEvaluator:
    def __init__(self):
        """Initialize with CLIP model"""
        print("Initializing DesignEdit Evaluator (SSIM + CLIP)...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        try:
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            print("✅ CLIP model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading CLIP model: {e}")
            sys.exit(1)
    
    def ssim_similarity(self, img1_path, img2_path):
        """
        METRIC 1: SSIM - Structural Similarity Index
        Measures precise layout/structure preservation
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
        Returns:
            float: SSIM score (0-1, higher = more structurally similar)
        """
        try:
            # Load images
            img1 = np.array(Image.open(img1_path).convert("RGB"))
            img2 = np.array(Image.open(img2_path).convert("RGB"))
            
            # Resize if needed
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            # Convert to grayscale for structural comparison
            gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
            
            # Compute SSIM
            ssim_score = metrics.structural_similarity(gray1, gray2, data_range=255)
            
            return max(0, min(1, ssim_score))  # Clamp to [0,1]
            
        except Exception as e:
            print(f"Error computing SSIM: {e}")
            return 0.0
    
    def clip_similarity(self, img1_path, img2_path):
        """
        METRIC 2: CLIP - Semantic Visual Similarity
        Measures content-aware visual similarity
        
        Args:
            img1_path: Path to first image
            img2_path: Path to second image
        Returns:
            float: CLIP cosine similarity (0-1, higher = more semantically similar)
        """
        try:
            with torch.no_grad():
                i1 = self.preprocess(Image.open(img1_path)).unsqueeze(0).to(self.device)
                i2 = self.preprocess(Image.open(img2_path)).unsqueeze(0).to(self.device)
                embs = self.model.encode_image(torch.cat([i1, i2]))
                similarity = torch.cosine_similarity(embs[0:1], embs[1:2]).item()
                
                # Normalize to [0,1] range (CLIP similarities can be negative)
                normalized_similarity = (similarity + 1) / 2
                return max(0, min(1, normalized_similarity))
                
        except Exception as e:
            print(f"Error computing CLIP similarity: {e}")
            return 0.0
    
    def evaluate_sample(self, original_img, modified_img):
        """
        Evaluate a single sample using both SSIM and CLIP
        
        Args:
            original_img: Path to original image
            modified_img: Path to modified image
            
        Returns:
            dict: Evaluation scores
        """
        ssim_score = self.ssim_similarity(original_img, modified_img)
        clip_score = self.clip_similarity(original_img, modified_img)
        
        # Combined score (you can adjust weights)
        combined_score = 0.5 * ssim_score + 0.5 * clip_score
        
        return {
            'ssim_similarity': ssim_score,
            'clip_similarity': clip_score,
            'combined_score': combined_score
        }

def load_sample_data(sample_dir):
    """
    Load sample data from a directory structure
    
    Expected structure:
    sample_dir/
    ├── original.html
    ├── original.png
    ├── instruction_1.txt
    ├── output_*.html (or modified_*.html)
    └── output_*.png (or modified_*.png)
    
    Returns:
        list: List of sample dictionaries
    """
    samples = []
    
    try:
        if not os.path.exists(sample_dir):
            print(f"Directory does not exist: {sample_dir}")
            return []
        
        # Find original image
        original_img = None
        for ext in ['png', 'jpg', 'jpeg']:
            img_path = os.path.join(sample_dir, f'original.{ext}')
            if os.path.exists(img_path):
                original_img = img_path
                break
        
        if not original_img:
            print(f"No original image found in {sample_dir}")
            return []
        
        # Find instruction files
        instruction_files = []
        for file in os.listdir(sample_dir):
            if file.startswith('instruction_') and (file.endswith('.txt') or file.endswith('.json')):
                instruction_files.append(file)
        
        if not instruction_files:
            print(f"No instruction files found in {sample_dir}")
            return []
        
        instruction_files.sort()
        
        # Process each instruction
        for inst_file in instruction_files:
            # Extract instruction number
            inst_num = inst_file.replace('instruction_', '').replace('.txt', '').replace('.json', '')
            
            # Load instruction text
            inst_path = os.path.join(sample_dir, inst_file)
            with open(inst_path, 'r', encoding='utf-8') as f:
                if inst_file.endswith('.json'):
                    instruction_data = json.load(f)
                    instruction = instruction_data.get('instruction', 
                                                    instruction_data.get('text', str(instruction_data)))
                else:
                    instruction = f.read().strip()
            
            # Find corresponding output image
            output_img = None
            output_img_patterns = [
                f'output_base_{inst_num}.png',
                f'gemini_output_{inst_num}.png',
                f'output_qwen_{inst_num}.png',
                f'output_gpt_{inst_num}.png',
                f'output_vl_base_{inst_num}.png',
                f'output_{inst_num}.png',
                f'modified_{inst_num}.png'
            ]
            
            for pattern in output_img_patterns:
                img_path = os.path.join(sample_dir, pattern)
                if os.path.exists(img_path):
                    output_img = img_path
                    break
            
            if not output_img:
                print(f"No output image found for instruction {inst_num} in {sample_dir}")
                continue
            
            # Create sample entry
            sample = {
                'sample_id': f"{os.path.basename(sample_dir)}_inst_{inst_num}",
                'instruction': instruction,
                'original_img': original_img,
                'modified_img': output_img
            }
            
            samples.append(sample)
        
        return samples
        
    except Exception as e:
        print(f"Error loading samples from {sample_dir}: {e}")
        traceback.print_exc()
        return []

def evaluate_single_sample(instruction, original_img_path, modified_img_path):
    """
    Evaluate a single sample and print results
    
    Args:
        instruction: Instruction text
        original_img_path: Path to original image
        modified_img_path: Path to modified image
    
    Returns:
        dict: Evaluation results
    """
    print("Evaluating single sample...")
    evaluator = DesignEditEvaluator()
    
    results = evaluator.evaluate_sample(original_img_path, modified_img_path)
    
    print(f"Instruction: {instruction}")
    print(f"SSIM Similarity (Structure): {results['ssim_similarity']:.3f}")
    print(f"CLIP Similarity (Semantic): {results['clip_similarity']:.3f}")
    print(f"Combined Score: {results['combined_score']:.3f}")
    
    return results

def evaluate_directory(directory_path, output_file=None):
    """
    Evaluate all samples in a directory
    
    Args:
        directory_path: Path to directory containing sample folders
        output_file: Optional output file path
        
    Returns:
        list: List of evaluation results
    """
    print(f"Evaluating directory: {directory_path}")
    
    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return []
    
    evaluator = DesignEditEvaluator()
    
    # Find all sample folders
    sample_folders = []
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isdir(item_path) and item.startswith('sample_'):
            sample_folders.append(item)
    
    sample_folders.sort()
    print(f"Found {len(sample_folders)} sample folders")
    
    # Load all samples
    all_samples = []
    for sample_folder in sample_folders:
        sample_path = os.path.join(directory_path, sample_folder)
        sample_data_list = load_sample_data(sample_path)
        
        if sample_data_list:
            all_samples.extend(sample_data_list)
            print(f"  Loaded {len(sample_data_list)} instruction(s) from {sample_folder}")
        else:
            print(f"  Failed to load {sample_folder}")
    
    if not all_samples:
        print("No valid samples found!")
        return []
    
    print(f"Total samples to evaluate: {len(all_samples)}")
    
    # Evaluate each sample
    all_results = []
    for i, sample in enumerate(all_samples):
        print(f"Evaluating {i+1}/{len(all_samples)}: {sample['sample_id']}")
        
        try:
            result = evaluator.evaluate_sample(
                sample['original_img'],
                sample['modified_img']
            )
            
            result['sample_id'] = sample['sample_id']
            result['instruction'] = sample['instruction']
            all_results.append(result)
            
            print(f"  SSIM: {result['ssim_similarity']:.3f}, CLIP: {result['clip_similarity']:.3f}, Combined: {result['combined_score']:.3f}")
            
        except Exception as e:
            print(f"  Error evaluating {sample['sample_id']}: {e}")
            continue
    
    # Print summary
    print(f"\n=== RESULTS SUMMARY FOR {directory_path} ===")
    print(f"Successfully evaluated: {len(all_results)} samples")
    
    if all_results:
        avg_ssim = sum(r['ssim_similarity'] for r in all_results) / len(all_results)
        avg_clip = sum(r['clip_similarity'] for r in all_results) / len(all_results)
        avg_combined = sum(r['combined_score'] for r in all_results) / len(all_results)
        print(f"Average SSIM (Structure): {avg_ssim:.3f}")
        print(f"Average CLIP (Semantic): {avg_clip:.3f}")
        print(f"Average Combined Score: {avg_combined:.3f}")
    
    # Save results to file
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== EVALUATION RESULTS FOR {directory_path} ===\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total samples: {len(all_results)}\n\n")
                
                if all_results:
                    f.write(f"Average SSIM (Structure): {avg_ssim:.3f}\n")
                    f.write(f"Average CLIP (Semantic): {avg_clip:.3f}\n")
                    f.write(f"Average Combined Score: {avg_combined:.3f}\n\n")
                
                f.write("Individual Results:\n")
                f.write("Sample ID\t\t\t\tSSIM\tCLIP\tCombined\tInstruction\n")
                f.write("-" * 120 + "\n")
                
                for result in all_results:
                    f.write(f"{result['sample_id']:<35}\t"
                           f"{result['ssim_similarity']:.3f}\t"
                           f"{result['clip_similarity']:.3f}\t"
                           f"{result['combined_score']:.3f}\t\t"
                           f"{result['instruction']}\n")
            
            print(f"Results saved to: {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    return all_results

def evaluate_all_directories(output_file="designedit_ssim_clip_results.txt"):
    """
    Evaluate all standard directories and save comprehensive results
    
    Args:
        output_file: Output file path
        
    Returns:
        dict: Results for each directory
    """
    # Standard directory names
    directories = [
        "evaluated_samples_qwen_tuned",
        "evaluated_samples_qwen_base",
        "evaluated_samples_qwen_vl", 
        "evaluated_samples_gemini",
        "evaluated_samples_gpt"
    ]
    
    all_directory_results = {}
    
    print("Starting comprehensive evaluation with SSIM + CLIP...")
    print(f"Results will be saved to: {output_file}")
    
    # Evaluate each directory
    for directory in directories:
        if os.path.exists(directory):
            print(f"\n{'='*50}")
            results = evaluate_directory(directory)
            if results:
                all_directory_results[directory] = results
        else:
            print(f"Directory not found: {directory}")
    
    # Save comprehensive results
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== DESIGNEDIT EVALUATION RESULTS (SSIM + CLIP) ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary table
            f.write("=== SUMMARY ===\n")
            f.write("Directory\t\t\tSamples\tSSIM\tCLIP\tCombined\n")
            f.write("-" * 70 + "\n")
            
            for dir_name, results in all_directory_results.items():
                if results:
                    avg_ssim = sum(r['ssim_similarity'] for r in results) / len(results)
                    avg_clip = sum(r['clip_similarity'] for r in results) / len(results)
                    avg_combined = sum(r['combined_score'] for r in results) / len(results)
                    f.write(f"{dir_name:<25}\t{len(results)}\t{avg_ssim:.3f}\t{avg_clip:.3f}\t{avg_combined:.3f}\n")
            
            f.write("\n" + "="*100 + "\n\n")
            
            # Detailed results for each directory
            for dir_name, results in all_directory_results.items():
                f.write(f"=== {dir_name} ===\n")
                f.write(f"Total samples: {len(results)}\n\n")
                
                f.write("Sample ID\t\t\t\tSSIM\tCLIP\tCombined\tInstruction\n")
                f.write("-" * 120 + "\n")
                
                for result in results:
                    f.write(f"{result['sample_id']:<35}\t"
                           f"{result['ssim_similarity']:.3f}\t"
                           f"{result['clip_similarity']:.3f}\t"
                           f"{result['combined_score']:.3f}\t\t"
                           f"{result['instruction'][:50]}...\n")
                
                f.write("\n")
        
        print(f"\n✅ Comprehensive results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving comprehensive results: {e}")
    
    return all_directory_results

def main():
    """Main function with example usage"""
    print("DesignEdit Evaluation Script - SSIM + CLIP")
    print("=" * 50)
    
    # Check if directories exist
    directories_to_check = [
        "evaluated_samples_qwen_tuned",
        "evaluated_samples_qwen_base",
        "evaluated_samples_qwen_vl", 
        "evaluated_samples_gemini",
        "evaluated_samples_gpt"
    ]
    
    existing_dirs = [d for d in directories_to_check if os.path.exists(d)]
    
    if not existing_dirs:
        print("No evaluation directories found!")
        print("Expected directories:")
        for d in directories_to_check:
            print(f"  - {d}")
        print("\nPlease ensure your data is in the correct directory structure.")
        return
    
    print(f"Found {len(existing_dirs)} directories to evaluate:")
    for d in existing_dirs:
        print(f"  ✅ {d}")
    
    # Run comprehensive evaluation
    print("\nStarting evaluation...")
    results = evaluate_all_directories("designedit_ssim_clip_results.txt")
    
    print(f"\n🎉 Evaluation complete!")
    print(f"Evaluated {sum(len(r) for r in results.values())} total samples")
    print("Check 'designedit_ssim_clip_results.txt' for detailed results")
    
    # Print key insights
    if results:
        print(f"\n=== KEY INSIGHTS ===")
        for dir_name, dir_results in results.items():
            if dir_results:
                avg_ssim = sum(r['ssim_similarity'] for r in dir_results) / len(dir_results)
                avg_clip = sum(r['clip_similarity'] for r in dir_results) / len(dir_results)
                print(f"{dir_name}:")
                print(f"  Structure Preservation (SSIM): {avg_ssim:.3f}")
                print(f"  Semantic Similarity (CLIP): {avg_clip:.3f}")

if __name__ == "__main__":
    main()