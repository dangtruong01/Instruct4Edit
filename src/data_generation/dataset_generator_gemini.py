import os
import json
import subprocess
from pathlib import Path
from datasets import load_dataset
from playwright.sync_api import sync_playwright
from itertools import islice
import google.generativeai as genai
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import requests

print(requests.get("https://ifconfig.me").text)


load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OUTPUT_DIR = "webcode2m_samples_gemini"

genai.configure(api_key=GOOGLE_API_KEY)  # NEW

# for model in genai.list_models():
#     print(model.name)

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.5-pro")
    response = model.generate_content(prompt)
    return response.text.strip()

def generate_instructions(html_code: str, few_shot_file: str = None) -> list:
    few_shot_examples = ""
    if few_shot_file and os.path.exists(few_shot_file):
        with open(few_shot_file, "r", encoding="utf-8") as f:
            few_shot_examples = f.read().strip() + "\n\n"

    prompt = (
        f"{few_shot_examples}"
        "You are a frontend UI expert. Given this HTML page, write 5 different, specific, visible design edit instructions as if you were giving them to a designer or developer in natural language.\n"
        "Instructions should be HUMAN-LIKE and must NOT mention or reference any code, class names, ids, or HTML tags directly.\n"
        "Focus on describing the desired visual change, layout, or style in plain English.\n"
        "Examples of human-like instructions:\n"
        "- Make the navigation bar background a darker color.\n"
        "- Add more space between the sections.\n"
        "- Center the main heading on the page.\n"
        "- Make all buttons rounded.\n"
        "- Hide the sidebar on mobile screens.\n"
        "Bad example (do NOT do this): 'Add border-radius: 5px to .container'\n\n"
        f"HTML:\n{html_code}\n\n"
        "Respond ONLY with the 5 instructions, numbered 1 to 5, and nothing else. Do not add explanations or code references.\n"
        "Instructions:\n"
        "1."
    )
    response = call_gemini(prompt).strip()
    # Parse the 5 instructions from the response
    instructions = []
    for line in response.split('\n'):
        if line.strip().startswith(tuple(str(i) + '.' for i in range(1, 6))):
            # Remove the number and dot, then strip
            instr = line.split('.', 1)[1].strip()
            if instr:
                instructions.append(instr)
    # If not enough instructions, try splitting by newlines
    if len(instructions) < 5:
        # fallback: split by lines and take first 5 non-empty
        instructions = [l.strip() for l in response.split('\n') if l.strip()]
        instructions = instructions[:5]
    return instructions

def is_instruction_faulty(instruction: str) -> bool:
    # Consider empty, whitespace, or too short/generic instructions as faulty
    if not instruction or not instruction.strip():
        return True
    if len(instruction.strip()) < 10:  # Arbitrary threshold for "too short"
        return True
    # Add more checks for generic/meaningless instructions if needed
    return False

def generate_modified_html(html_code: str, instruction: str) -> str:
    prompt = f"""You are a frontend developer. Given the HTML code and a design instruction, apply the change and return the full modified HTML file. 
Do NOT skip any parts of the code — output the complete modified HTML with only the necessary edits.

Instruction:
{instruction}

Original HTML:
{html_code}

Modified HTML:"""
    return call_gemini(prompt)

def clean_html_code(html: str) -> str:
    html = html.strip()
    # Remove ```html or ``` at the start
    if html.startswith("```html"):
        html = html[7:].lstrip()
    elif html.startswith("```"):
        html = html[3:].lstrip()
    # Remove ``` at the end
    if html.endswith("```"):
        html = html[:-3].rstrip()
    return html

def capture_screenshot(html_path: Path, screenshot_path: Path, viewport_size=(1280, 720)):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": viewport_size[0], "height": viewport_size[1]})
        page.goto(f"file://{html_path.resolve()}")
        page.screenshot(path=str(screenshot_path), full_page=True)
        browser.close()

def save_sample(
    sample_id: str,
    original_html: str,
    instructions: list,
    modified_htmls: list,
    image=None
):
    base_path = Path(OUTPUT_DIR) / f"sample_{sample_id}"
    base_path.mkdir(parents=True, exist_ok=True)
    original_html_path = base_path / "original.html"
    original_html_path.write_text(original_html, encoding="utf-8")

    # Always generate the original.png using Playwright for consistency
    original_png_path = base_path / "original.png"
    capture_screenshot(original_html_path, original_png_path)  # Use same viewport as for modified
    
    # Save each instruction and modified HTML
    for idx, (instruction, modified_html) in enumerate(zip(instructions, modified_htmls)):
        (base_path / f"instruction_{idx+1}.txt").write_text(instruction, encoding="utf-8")
        (base_path / f"modified_{idx+1}.html").write_text(clean_html_code(modified_html), encoding="utf-8")
    return base_path

def verify_instruction_applied(ori_image_path, after_image_path, instruction):
    # Guard against faulty instructions
    if is_instruction_faulty(instruction):
        return (
            "❌ Not Applied\n\n"
            "The instruction was empty, too short, or invalid. No meaningful verification can be performed."
        )

    with open(ori_image_path, "rb") as f:
        ori_image_bytes = f.read()
    with open(after_image_path, "rb") as f:
        after_image_bytes = f.read()

    prompt = (
        "You are a highly cautious visual UI/UX reviewer. I will show you two images: one \"before\" and one \"after\" a design change was instructed.\n\n"
        "- The FIRST image is the ORIGINAL (before the design change).\n"
        "- The SECOND image is the MODIFIED (after the design change was applied).\n\n"
        "Your task:\n"
        "Compare the two images and verify whether the following instruction was fully and clearly implemented in the modified image.\n"
        "If you're unsure, highlight what might be missing or only partially done.\n\n"
        "Instruction:\n"
        f"{instruction}\n\n"
        "Step-by-step justification:\n"
        "1. What is the intended visual change?\n"
        "2. Is this change clearly visible in the after image compared to the before image?\n"
        "3. Are there any doubts, ambiguities, or missing elements?\n\n"
        "Final verdict (choose only one):\n"
        "- ✅ Fully Applied\n"
        # "- ⚠️ Partially Applied (explain what’s incomplete or vague)\n"
        "- ❌ Not Applied\n\n"
        "Now compare the two images and respond accordingly.\n"
        "Only take VISUAL changes into consideration."
    )

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        [
            prompt,
            {"mime_type": "image/png", "data": ori_image_bytes},
            {"mime_type": "image/png", "data": after_image_bytes},
        ]
    )
    return response.text.strip()

def process_sample(index: int, html_code: str, image=None):
    try:
        instructions = generate_instructions(html_code)
        modified_htmls = []
        verification_results = []
        base_path = None

        for i, instruction in enumerate(instructions):
            modified_html = generate_modified_html(html_code, instruction)
            modified_htmls.append(modified_html)

        base_path = save_sample(index, html_code, instructions, modified_htmls, image=image)

        # Capture screenshots for each modified HTML and run verification
        for i in range(len(modified_htmls)):
            try:
                html_file = base_path / f"modified_{i+1}.html"
                screenshot_file = base_path / f"screenshot_{i+1}.png"
                capture_screenshot(html_file, screenshot_file)

                # Run image verification (original.png vs. screenshot_{i+1}.png)
                ori_image_path = base_path / "original.png"
                after_image_path = screenshot_file
                instruction = instructions[i]
                verification = verify_instruction_applied(ori_image_path, after_image_path, instruction)
                verification_file = base_path / f"verification_{i+1}.txt"
                verification_file.write_text(verification, encoding="utf-8")
                verification_results.append(verification)

            except Exception as inner_e:
                print(f"[!] Failed to process instruction {i+1} in sample {index}: {inner_e}")
                continue
        print(f"[✓] Processed sample {index}")
    except Exception as e:
        print(f"[✗] Failed sample {index}: {e}")

def run_webcode2m_subset(num_samples=10, start_idx=0, max_workers=5):
    dataset = load_dataset("xcodemind/webcode2m", split="train", streaming=True)
    samples = list(islice(dataset, start_idx, start_idx + num_samples))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, sample in enumerate(samples):
            html_code = sample["text"]
            image = sample.get("image", None)
            futures.append(executor.submit(process_sample, idx + start_idx, html_code, image))

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"[!] Exception in thread: {e}")

def regenerate_verifications(output_dir=OUTPUT_DIR, start_sample=0):
    """
    Iterate through each sample folder and regenerate verification files using verify_instruction_applied.
    """
    base_dir = Path(output_dir)
    for sample_dir in sorted(base_dir.glob("sample_*")):
        # Extract the sample index from the folder name
        try:
            sample_idx = int(sample_dir.name.replace("sample_", ""))
        except ValueError:
            continue
        if sample_idx < start_sample:
            continue

        print(f"Processing {sample_dir.name}...")
        original_img = sample_dir / "original.png"
        for i in range(1, 6):  # Adjust range if you expect more than 5 instructions per sample
            mod_html = sample_dir / f"modified_{i}.html"
            screenshot = sample_dir / f"screenshot_{i}.png"
            instruction_file = sample_dir / f"instruction_{i}.txt"
            verification_file = sample_dir / f"verification_{i}.txt"
            if mod_html.exists() and screenshot.exists() and instruction_file.exists() and original_img.exists():
                instruction = instruction_file.read_text(encoding="utf-8")
                # Optionally regenerate screenshot here if needed:
                # capture_screenshot(mod_html, screenshot)
                verification = verify_instruction_applied(original_img, screenshot, instruction)
                verification_file.write_text(verification, encoding="utf-8")
                print(f"[✓] Regenerated {verification_file}")
            else:
                print(f"  Missing: {', '.join(str(f) for f in [mod_html, screenshot, instruction_file, original_img] if not f.exists())}")
                continue


if __name__ == "__main__":
    start_time = time.time()
    run_webcode2m_subset(start_idx =726, num_samples=50, max_workers=1)  # Modify to process more
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"\nTotal runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")