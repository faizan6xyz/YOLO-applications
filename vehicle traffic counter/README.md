# 🚗 Real-Time Vehicle & Traffic Counter

Detect and count cars, bikes, and pedestrians from a traffic video or webcam feed using YOLOv8.

## Demo
> Add a `demo.gif` here showing live detection

## Features
- Multi-class vehicle detection (car, truck, bike, pedestrian)
- Entry/exit line counting logic
- Live FPS and count dashboard
- Export counts to CSV per minute

## Setup
```bash
pip install -r requirements.txt
python app.py --source video.mp4
# or use webcam:
python app.py --source 0
```

## How It Works
Uses YOLOv8 with OpenCV to process each frame, draws a counting line, and logs every object that crosses it.

## Tech Stack
- YOLOv8 (Ultralytics)
- OpenCV
- Python
- Streamlit (dashboard)
