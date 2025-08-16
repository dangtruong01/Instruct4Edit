import os
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
import time

def find_html_files(root_dir):
    html_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('v2.html'):
                html_files.append(os.path.join(root, file))
    return html_files

def html_to_png(html_path, png_path):
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1280,1024")
    driver = webdriver.Chrome(options=options)
    driver.get('file://' + os.path.abspath(html_path))
    time.sleep(1)  # Wait for rendering
    driver.save_screenshot(png_path)
    driver.quit()

    # Optionally crop whitespace (optional, can be removed)
    img = Image.open(png_path)
    img = img.crop(img.getbbox())
    img.save(png_path)

if __name__ == "__main__":
    root_dir = "."
    html_files = find_html_files(root_dir)
    for html_file in html_files:
        png_file = os.path.splitext(html_file)[0] + ".png"
        print(f"Converting {html_file} -> {png_file}")
        html_to_png(html_file, png_file)