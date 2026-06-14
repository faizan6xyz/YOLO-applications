"""
FastAPI endpoint: POST /detect with an image file, returns JSON detections.
"""
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io, numpy as np

app = FastAPI(title="PPE Detector API")
model = YOLO("yolov8n.pt")  # replace with fine-tuned weights
VIOLATION_CLASSES = ["no-helmet", "no-vest"]

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img)

    results = model(img_np, verbose=False)[0]
    detections = []

    for box in results.boxes:
        label = model.names[int(box.cls)]
        detections.append({
            "label": label,
            "confidence": round(float(box.conf), 3),
            "bbox": list(map(int, box.xyxy[0])),
            "violation": label in VIOLATION_CLASSES,
        })

    return {
        "total_detections": len(detections),
        "violations": [d for d in detections if d["violation"]],
        "detections": detections,
    }
