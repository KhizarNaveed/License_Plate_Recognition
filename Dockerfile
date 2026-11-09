# Use official lightweight Python 3.10 runtime
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by OpenCV and EasyOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download EasyOCR English detection & recognition models into image cache
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)"

# Copy project code, sample images, and trained YOLOv8 weights
COPY Final_code/ ./Final_code/
COPY sample_images/ ./sample_images/
COPY runs/detect/train_final/weights/best.pt ./runs/detect/train_final/weights/best.pt
COPY data.yaml custom_hyp.yaml ./

# Create output directory
RUN mkdir -p /app/cropped_output

# Run the ALPR inference pipeline by default
CMD ["python", "Final_code/charset.py"]

