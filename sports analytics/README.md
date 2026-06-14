# ⚽ Sports Analytics with YOLO + Pose Estimation

Track athletes in a game video, detect player positions, and overlay speed, distance traveled, and formation analysis using YOLOv8 + ByteTrack.

## Demo
> Add a `demo.gif` here

## Features
- Multi-player tracking with unique persistent IDs (ByteTrack)
- Speed & distance estimation per player
- Pose keypoints for movement/form analysis
- Bird's-eye view team formation map
- Works on football, basketball, cricket footage

## Setup
```bash
pip install -r requirements.txt
python tracker.py --source match_clip.mp4
python pose.py --source match_clip.mp4
```

## How It Works
1. YOLOv8 detects players each frame
2. ByteTrack assigns consistent IDs across frames
3. Displacement between frames → speed estimation
4. YOLOv8-pose adds 17 keypoints per player
5. Top-down view generated using homography transform

## Tech Stack
- YOLOv8 + YOLOv8-pose (Ultralytics)
- ByteTrack (via supervision)
- supervision library
- NumPy / OpenCV
