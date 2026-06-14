"""
Pose estimation overlay for player movement analysis.
Uses YOLOv8-pose to extract 17 COCO keypoints per player.
"""
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

POSE_MODEL = "yolov8n-pose.pt"

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),      # head
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms
    (5, 11), (6, 12), (11, 12),            # torso
    (11, 13), (13, 15), (12, 14), (14, 16) # legs
]

def draw_pose(frame, keypoints, color=(0, 255, 180)):
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.5:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1)
    for a, b in SKELETON:
        if keypoints[a][2] > 0.5 and keypoints[b][2] > 0.5:
            pt1 = (int(keypoints[a][0]), int(keypoints[a][1]))
            pt2 = (int(keypoints[b][0]), int(keypoints[b][1]))
            cv2.line(frame, pt1, pt2, color, 2)

def run(source: str):
    model = YOLO(POSE_MODEL)
    cap = cv2.VideoCapture(source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        if results.keypoints is not None:
            for kps in results.keypoints.data:
                draw_pose(frame, kps.numpy())

        cv2.imshow("Pose Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    run(args.source)
