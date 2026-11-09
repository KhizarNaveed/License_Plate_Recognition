import os
import re
import easyocr

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cropped_dir = os.path.join(BASE_DIR, 'runs', 'detect', 'cropped_plates')
reader = easyocr.Reader(['en'])

# === Correction rules (common OCR mistakes) ===
ocr_corrections = {
    'O': '0',
    'I': '1',
    "'": '-',   # e.g., 65'E1 → 65-E1
    ' ': '-'    # spaces to dashes
}

# === Valid license plate regex (Vietnam-style or similar)
regex_pattern = r"^\d{2}[-A-Z]{1,2}\d{0,4}$|^\d{4}$|^\d{2}-[A-Z0-9]{1,3}$"

# === Optional Ground Truth ===
ground_truth = {
    "0790": "0790",
    "65'E1": "65-E1",
    "66 M": "66-M",
    "6766": "6766",
    "5502": "5502",
    "4012": "4012",
    "0206": "0206"
}

print("🚀 Starting OCR and plate validation...\n")

# === Begin processing cropped images ===
for root, dirs, files in os.walk(cropped_dir):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, filename)
            print(f"\n📄 Image: {filename}")

            results = reader.readtext(image_path)

            if results:
                for bbox, text, confidence in results:
                    print(f"🔍 Raw OCR: {text} | Confidence: {confidence:.2f}")

                    if confidence < 0.70:
                        print("   ❌ Skipped (low confidence)")
                        continue

                    # Apply common OCR corrections
                    cleaned_text = text
                    for wrong, right in ocr_corrections.items():
                        cleaned_text = cleaned_text.replace(wrong, right)

                    # Validate plate format
                    if re.match(regex_pattern, cleaned_text):
                        print(f"   ✅ Valid Plate: {cleaned_text}")
                    else:
                        print(f"   ⚠️ Format Suspect: {cleaned_text}")

                    # Optional: Ground truth validation
                    if text in ground_truth:
                        expected = ground_truth[text]
                        if cleaned_text == expected:
                            print("   🔍 Ground Truth Match ✅")
                        else:
                            print(f"   ❗ Ground Truth Mismatch ❌ → Expected: {expected}")
            else:
                print("   ❌ No text detected.")

print("\n✅ Processing Complete.")
