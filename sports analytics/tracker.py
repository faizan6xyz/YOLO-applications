"""
Multi-player tracker with speed and distance estimation.
Uses YOLOv8 + ByteTrack via the supervision library.
"""
import cv2
import argparse
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

MODEL = "yolov8n.pt"
FPS = 30
METERS_PER_PIXEL = 0.05  # tune based on camera angle

def estimate_speed(positions: list, fps: int, mpp: float) -> float:
    """Returns speed in km/h given pixel positions over time."""
    if len(positions) < 2:
        return 0.0
    dx = positions[-1][0] - positions[-2][0]
    dy = positions[-1][1] - positions[-2][1]
    pixel_dist = np.sqrt(dx**2 + dy**2)
    meters = pixel_dist * mpp
    speed_ms = meters * fps
    return round(speed_ms * 3.6, 1)

def run(source: str):
    model = YOLO(MODEL)
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    positions = defaultdict(list)
    distances = defaultdict(float)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = model(frame, classes=[0], verbose=False)[0]  # class 0 = person
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = []
        for det_idx, (xyxy, _, conf, cls, track_id, _) in enumerate(detections):
            if track_id is None:
                continue
            cx = int((xyxy[0] + xyxy[2]) / 2)
            cy = int((xyxy[1] + xyxy[3]) / 2)
            positions[track_id].append((cx, cy))

            if len(positions[track_id]) > 1:
                dx = positions[track_id][-1][0] - positions[track_id][-2][0]
                dy = positions[track_id][-1][1] - positions[track_id][-2][1]
                distances[track_id] += np.sqrt(dx**2 + dy**2) * METERS_PER_PIXEL

            speed = estimate_speed(positions[track_id], fps, METERS_PER_PIXEL)
            dist = round(distances[track_id], 1)
            labels.append(f"#{track_id} {speed}km/h {dist}m")

        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, labels=labels)

        cv2.imshow("Sports Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Tracking complete.")
    for tid, dist in distances.items():
        print(f"  Player #{tid}: {round(dist, 1)}m covered")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    run(args.source)
