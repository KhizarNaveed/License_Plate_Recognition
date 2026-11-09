import os
import cv2

# --- Configuration ---
images_dir = r"C:\Users\Sanani\Desktop\Six Semester\Machine learning\License Plate Recognition.v11i.yolov8\runs\detect\predict"
labels_dir = r"C:\Users\Sanani\Desktop\Six Semester\Machine learning\License Plate Recognition.v11i.yolov8\runs\detect\predict\labels"
output_dir = r"C:\Users\Sanani\Desktop\Six Semester\Machine learning\License Plate Recognition.v11i.yolov8\runs\detect\cropped_plates"

os.makedirs(output_dir, exist_ok=True)

# --- Process Each Image ---
for filename in os.listdir(images_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(images_dir, filename)
        label_path = os.path.join(labels_dir, filename.replace(".jpg", ".txt").replace(".png", ".txt"))
        
        if not os.path.exists(label_path):
            print(f"[!] Label file not found for {filename}, skipping.")
            continue
        
        img = cv2.imread(image_path)
        h, w, _ = img.shape

        with open(label_path, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)

                # Convert from normalized to pixel
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)

                # Clamp values to image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)

                cropped = img[y1:y2, x1:x2]
                out_name = f"{os.path.splitext(filename)[0]}_plate{idx}.jpg"
                cv2.imwrite(os.path.join(output_dir, out_name), cropped)
                print(f"[✓] Saved cropped: {out_name}")
