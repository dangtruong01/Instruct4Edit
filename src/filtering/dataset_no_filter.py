import json
from pathlib import Path

def collect_accepted_cases(dataset_dir, output_json):
    dataset_dir = Path(dataset_dir)
    results = []
    counter = 1  # Start from 1 for sequential sample IDs
    for sample_dir in sorted(dataset_dir.glob("sample_*")):
        original_html_path = sample_dir / "original.html"
        if not original_html_path.exists():
            continue
        original_html = original_html_path.read_text(encoding="utf-8")
        for i in range(1, 6):  # Adjust if you have more than 5 instructions per sample
            instruction_file = sample_dir / f"instruction_{i}.txt"
            modified_html_file = sample_dir / f"modified_{i}.html"
            verification_file = sample_dir / f"verification_{i}.txt"
            if not (instruction_file.exists() and modified_html_file.exists() and verification_file.exists()):
                continue
            instruction = instruction_file.read_text(encoding="utf-8").strip()
            modified_html = modified_html_file.read_text(encoding="utf-8").strip()
            sample_id = f"sample_{counter}"
            results.append({
                "id": sample_id,
                "instruction": instruction,
                "original_html": original_html,
                "modified_html": modified_html
            })
            counter += 1
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} accepted cases to {output_json}")

# Example usage, update the file paths:
collect_accepted_cases("webcode2m_samples_gemini", "unfiltered_instruction_tuning_data.json")