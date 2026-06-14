# 🦺 PPE Safety Compliance Detector

Detect whether workers are wearing helmets, safety vests, and gloves using a fine-tuned YOLOv8 model. Raises real-time alerts for violations.

## Demo
> Add a `demo.gif` here

## Features
- Detects: helmet, no-helmet, vest, no-vest, gloves
- Real-time violation alerts with timestamps
- REST API via FastAPI for integration with cameras
- Custom dataset fine-tuned via Roboflow

## Dataset
Use the public PPE dataset from Roboflow Universe:
https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety

## Setup
```bash
pip install -r requirements.txt

# Fine-tune the model (optional, uses pretrained weights by default)
python train.py

# Run detection on video
python detect.py --source site_video.mp4

# Start API server
uvicorn api:app --reload
```

## API Usage
```bash
curl -X POST "http://localhost:8000/detect" \
  -F "file=@worker_image.jpg"
```

## Tech Stack
- YOLOv8 (Ultralytics)
- Roboflow (dataset management)
- FastAPI
- Python
