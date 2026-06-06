from ultralytics import YOLO
import cv2
import os
model = YOLO("best.pt")
video_path = "helmet_video.mp4"
cap = cv2.VideoCapture(video_path)
os.makedirs("detected_frames", exist_ok=True)
frame_count = 0
saved_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    detected = False
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            if model.names[class_id] == "no_helmet":
                detected = True
                break
    if detected:
        cv2.imwrite(
            f"detected_frames/frame_{saved_count}.jpg",frame)
        saved_count += 1
    frame_count += 1
cap.release()
print(f"Saved {saved_count} frames containing no_helmet")