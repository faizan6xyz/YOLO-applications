import cv2
import argparse
from ultralytics import YOLO
from datetime import datetime

VIOLATION_CLASSES = ["no-helmet", "no-vest"]  # adjust to your dataset labels
MODEL_PATH = "runs/train/ppe_detector/weights/best.pt"  # or "yolov8n.pt" for demo

def run(source):
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(source if not source.isdigit() else int(source))
    violations_log = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            is_violation = label in VIOLATION_CLASSES
            color = (0, 0, 255) if is_violation else (0, 200, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if is_violation:
                ts = datetime.now().isoformat()
                violations_log.append({"time": ts, "violation": label, "confidence": round(conf, 3)})
                cv2.putText(frame, "⚠ VIOLATION", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("PPE Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if violations_log:
        import pandas as pd
        pd.DataFrame(violations_log).to_csv("violations_log.csv", index=False)
        print(f"{len(violations_log)} violations logged to violations_log.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args = parser.parse_args()
    run(args.source)
