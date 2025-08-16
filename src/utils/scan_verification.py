import os
from pathlib import Path

def scan_verification_files(base_dir):
    base_dir = Path(base_dir)
    total = 0
    fully_applied = 0
    partially_applied = 0
    not_applied = 0

    for sample_dir in base_dir.glob("sample_*"):
        for vf in sample_dir.glob("verification_*.txt"):
            total += 1
            content = vf.read_text(encoding="utf-8").lower()
            if "✅ fully applied" in content or "fully applied" in content:
                fully_applied += 1
            elif "⚠️ partially applied" in content:
                partially_applied += 1
            elif "❌ not applied" in content or "not applied" in content:
                not_applied += 1

    accepted = fully_applied
    not_accepted = partially_applied + not_applied
    acceptance_rate = (accepted / total) * 100 if total > 0 else 0

    print(f"Total verifications: {total}")
    print(f"Fully Applied: {fully_applied}")
    print(f"Partially Applied: {partially_applied}")
    print(f"Not Applied: {not_applied}")
    print(f"Accepted (Fully Applied): {accepted}")
    print(f"Not Accepted (Partial/Not Applied): {not_accepted}")
    print(f"Acceptance Rate: {acceptance_rate:.2f}%")

if __name__ == "__main__":
    scan_verification_files("webcode2m_samples_gemini_v9")