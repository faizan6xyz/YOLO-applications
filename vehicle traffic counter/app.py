import cv2
import argparse
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# --- Config ---
MODEL_PATH = "yolov8n.pt"   # auto-downloads on first run
LINE_Y = 400                 # horizontal counting line position
CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck", 0: "person"}

def draw_line(frame, y):
    cv2.line(frame, (0, y), (frame.shape[1], y), (0, 255, 255), 2)

def main(source):
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(source if source.isdigit() is False else int(source))

    counts = {label: 0 for label in CLASSES.values()}
    tracked_ids = set()
    log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=list(CLASSES.keys()), verbose=False)

        if results[0].boxes.id is not None:
            for box, cls_id, track_id in zip(
                results[0].boxes.xyxy,
                results[0].boxes.cls,
                results[0].boxes.id
            ):
                x1, y1, x2, y2 = map(int, box)
                cy = (y1 + y2) // 2
                label = CLASSES.get(int(cls_id), "unknown")
                tid = int(track_id)

                # Count when crossing the line
                if abs(cy - LINE_Y) < 10 and tid not in tracked_ids:
                    tracked_ids.add(tid)
                    counts[label] = counts.get(label, 0) + 1
                    log.append({"time": datetime.now().isoformat(), "class": label, "id": tid})

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.putText(frame, f"{label} #{tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)

        draw_line(frame, LINE_Y)

        # Overlay counts
        y_offset = 30
        for label, count in counts.items():
            cv2.putText(frame, f"{label}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

        cv2.imshow("Traffic Counter", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save log
    df = pd.DataFrame(log)
    if not df.empty:
        df.to_csv("counts_log.csv", index=False)
        print(f"Saved {len(df)} detections to counts_log.csv")
    print("Final counts:", counts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Video path or webcam index")
    args = parser.parse_args()
    main(args.source)
