from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import re

images = convert_from_path("HSC26_Bangla_1st_paper.pdf", dpi=300)
for i, img in enumerate(images):
    img.save(f"page_{i+1}.png")


def ocr_bangla(image_path):
    return pytesseract.image_to_string(Image.open(image_path), lang='ben')

text = ""
for i in range(len(images)):
    text += ocr_bangla(f"page_{i+1}.png") + "\n"


def clean_bangla_text(text):
  
    text = re.sub(r'[^\u0980-\u09FF।,!?০-৯\s]', '', text)  # Keep only Bengali characters
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([।!?])', r'\1\n', text)  # Add newline after sentence ends
    return text.strip()

# Clean the extracted text
cleaned_text = clean_bangla_text(text)

# Save the cleaned text to pdf_text.txt
with open("pdf_text.txt", "w", encoding="utf-8") as f:
    f.write(cleaned_text)

print(f"Text extraction completed. Saved to pdf_text.txt")
print(f"Total characters: {len(cleaned_text)}")
print(f"Total pages processed: {len(images)}")
