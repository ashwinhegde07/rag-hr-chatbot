import json
import re
from PyPDF2 import PdfReader
from pathlib import Path

def extract_pdf_text(pdf_path: str) -> str:
    """Extract all text from PDF."""
    reader = PdfReader(pdf_path)
    text = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def clean_text(text: str) -> str:
    """Remove unwanted characters and normalize spacing."""
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces/newlines
    text = re.sub(r'[^A-Za-z0-9.,;:\-()&%\s]', '', text)  # remove non-text chars
    return text.strip()

def save_to_json(text: str, output_path: str):
    """Save cleaned text to JSON file."""
    data = {"content": text}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    input_pdf = Path("data/HR-Policy.pdf")
    output_json = Path("data/hr_text.json")

    if not input_pdf.exists():
        raise FileNotFoundError(f"{input_pdf} not found.")

    print("Extracting text...")
    raw_text = extract_pdf_text(str(input_pdf))

    print("Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("Saving to JSON...")
    save_to_json(cleaned_text, str(output_json))

    print("✅ Done. Cleaned data saved at:", output_json)
