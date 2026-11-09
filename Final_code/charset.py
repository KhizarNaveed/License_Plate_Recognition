from ultralytics import YOLO
import cv2
import easyocr
import os

# --- Initialize OCR reader ---
reader = easyocr.Reader(['en'])  # Initialize EasyOCR for English language

def recognize_plates_from_image(image_path, model_path, output_dir, show_ocr_overlay=True): # Function to recognize license plates from an image
    model = YOLO(model_path) # Load the YOLO model
    img = cv2.imread(image_path) # Read the image
    h, w, _ = img.shape # Get image dimensions needed for cropping

    results = model(image_path) # Run inference on the image Result contain bounding boxes
    os.makedirs(output_dir, exist_ok=True)  # Create output directory(cropped_output) if it doesn't exist

    detected_any = False # Flag to check if any plates were detected
    PAD = 5  # Padding around bounding box why padding?  To ensure the license plate is fully captured + due to OCR sometimes needing a bit more space to read the text.

    for idx, box in enumerate(results[0].boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get bounding box coordinates x1, y1, x2, y2
        x1 = max(0, x1 - PAD)   # Ensure x1 is not negative
        y1 = max(0, y1 - PAD)   # Ensure y1 is not negative
        x2 = min(w, x2 + PAD)   # Ensure x2 is not out of bounds
        y2 = min(h, y2 + PAD)   # Ensure y2 is not out of bounds

        plate_crop = img[y1:y2, x1:x2]      # Crop the plate area
        crop_name = f"plate_crop_{idx}.jpg"             # Name of the cropped image 
        crop_path = os.path.join(output_dir, crop_name)     
        cv2.imwrite(crop_path, plate_crop)      # image write to the path folder
        print(f"[✓] Saved cropped plate: {crop_path}")   # check if the image is saved

        # Convert to RGB for OCR
        plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)

        # Run OCR
        ocr_results = reader.readtext(plate_rgb) 

        if ocr_results:
            # Filter out low-confidence or short junk text
            filtered = [r for r in ocr_results if r[2] > 0.5 and len(r[1]) >= 2]   # Confidence threshold and text length check

            # Sort left-to-right
            filtered.sort(key=lambda r: r[0][0][0]) # arrange them by x-coordinate from left to right as in license plate

            if filtered:    # "Did we get any good OCR results after filtering?"
                # Combine all detected parts
                combined_text = ' '.join([r[1] for r in filtered])
                print(f"[🔍] Detected Plate {idx + 1}: {combined_text}")
            else:
                print("[!] OCR text found, but all filtered out.") # “OCR found something, but it was junk — so we removed all.”

            # Optional: draw overlays
            if show_ocr_overlay:
                for bbox, text, conf in filtered:
                    pts = [(int(p[0]), int(p[1])) for p in bbox]
                    for i in range(4):  # Drawing the green box
                        cv2.line(plate_crop, pts[i], pts[(i + 1) % 4], (0, 255, 0), 2)
                    cv2.putText(plate_crop, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                overlay_name = f"plate_ocr_overlay_{idx}.jpg"
                overlay_path = os.path.join(output_dir, overlay_name)
                cv2.imwrite(overlay_path, plate_crop)
                print(f"[🖼️] Saved OCR overlay: {overlay_path}")
        else:
            print("[!] OCR failed to detect text.")

        detected_any = True

    if not detected_any:
        print("[!] No license plate detected.")

# --- Set Paths ---
image_path = r"D:\Six Semester\Machine learning\License Plate Recognition.v11i.yolov8\double.jpeg"
model_path = r"D:\Six Semester\Machine learning\License Plate Recognition.v11i.yolov8\runs\detect\train_final\weights\best.pt"
output_dir = r"D:\Six Semester\Machine learning\License Plate Recognition.v11i.yolov8\cropped_output"

# --- Run the Code ---
recognize_plates_from_image(image_path, model_path, output_dir)
