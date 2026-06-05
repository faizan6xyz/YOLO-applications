"""
Helmet Detection System
- Detects persons and helmets using YOLOv8
- Captures photos of violators (no helmet)
- Saves violations with timestamp
"""

import cv2
import os
import time
import json
from datetime import datetime
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] ultralytics not installed. Run: pip install ultralytics")

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
CONFIG = {
    "model_path": "yolov8n.pt",          # Base YOLO model (auto-downloads)
    "helmet_model_path": None,           # Set to custom helmet model if available
    "capture_dir": "captures",           # Folder to save violation photos
    "confidence_threshold": 0.5,         # Minimum detection confidence
    "cooldown_seconds": 5,               # Seconds between captures of same person
    "camera_index": 0,                   # 0 = default webcam
    "video_source": None,                # Set to video file path to use video instead
    "frame_width": 1280,
    "frame_height": 720,
    "log_file": "violations_log.json",
}

# COCO class IDs
PERSON_CLASS_ID = 0
# If using custom helmet model, set these:
HELMET_CLASS_ID = 0     # 'helmet' in custom model
NO_HELMET_CLASS_ID = 1  # 'no_helmet' in custom model


# ─────────────────────────────────────────────
# Violation Logger
# ─────────────────────────────────────────────
class ViolationLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.violations = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                self.violations = json.load(f)

    def log(self, photo_path, confidence):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "photo": photo_path,
            "confidence": round(confidence, 3),
        }
        self.violations.append(entry)
        with open(self.log_file, 'w') as f:
            json.dump(self.violations, f, indent=2)
        return entry

    def total(self):
        return len(self.violations)


# ─────────────────────────────────────────────
# Helmet Detector
# ─────────────────────────────────────────────
class HelmetDetector:
    def __init__(self, config=CONFIG):
        self.config = config
        self.capture_dir = Path(config["capture_dir"])
        self.capture_dir.mkdir(exist_ok=True)
        self.logger = ViolationLogger(config["log_file"])
        self.last_capture_time = {}  # track cooldown per region
        self.violation_count = 0

        if YOLO_AVAILABLE:
            # Load custom helmet model if provided, else use base YOLOv8
            model_path = config.get("helmet_model_path") or config["model_path"]
            print(f"[INFO] Loading model: {model_path}")
            self.model = YOLO(model_path)
            self.use_custom_model = config.get("helmet_model_path") is not None
        else:
            self.model = None
            self.use_custom_model = False

    def detect(self, frame):
        """
        Run detection on a frame.
        Returns list of violators: [{"bbox": (x1,y1,x2,y2), "confidence": float}]
        """
        if self.model is None:
            return []

        results = self.model(frame, conf=self.config["confidence_threshold"], verbose=False)
        violators = []

        if self.use_custom_model:
            # Custom model: directly detects 'helmet' vs 'no_helmet'
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == NO_HELMET_CLASS_ID:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        violators.append({"bbox": (x1, y1, x2, y2), "confidence": conf})
        else:
            # Base model: detect persons, then check for helmet overlap
            persons = []
            helmets = []

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    if cls_id == PERSON_CLASS_ID:
                        persons.append({"bbox": (x1, y1, x2, y2), "confidence": conf})

            # Since base YOLOv8 doesn't have 'helmet', we simulate helmet region
            # as the upper 25% of the person's bounding box (head area)
            for person in persons:
                x1, y1, x2, y2 = person["bbox"]
                head_y2 = y1 + int((y2 - y1) * 0.25)
                head_region = (x1, y1, x2, head_y2)

                has_helmet = self._check_helmet_in_region(frame, head_region)

                if not has_helmet:
                    violators.append({
                        "bbox": person["bbox"],
                        "confidence": person["confidence"]
                    })

        return violators

    def _check_helmet_in_region(self, frame, region):
        """
        Simple brightness/color heuristic for helmet detection in head region.
        In production, replace with a trained helmet classifier.
        """
        x1, y1, x2, y2 = region
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return False

        head_crop = frame[y1:y2, x1:x2]
        if head_crop.size == 0:
            return False

        # Convert to HSV and check for typical helmet colors
        # (hard hats are usually yellow, white, orange, red, blue)
        hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)

        # Yellow helmet
        yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
        # White helmet
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        # Orange helmet
        orange_mask = cv2.inRange(hsv, (5, 100, 100), (20, 255, 255))
        # Red helmet
        red_mask1 = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
        red_mask2 = cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        # Blue helmet
        blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))

        combined = (yellow_mask | white_mask | orange_mask |
                    red_mask1 | red_mask2 | blue_mask)

        # If more than 15% of head area matches helmet colors → has helmet
        coverage = cv2.countNonZero(combined) / (head_crop.shape[0] * head_crop.shape[1] + 1e-5)
        return coverage > 0.15

    def capture_violation(self, frame, bbox, confidence):
        """Save a violation photo with bounding box drawn."""
        now = time.time()

        # Cooldown: avoid duplicate captures for same region
        region_key = f"{bbox[0]//50}_{bbox[1]//50}"  # grid cell
        if region_key in self.last_capture_time:
            if now - self.last_capture_time[region_key] < self.config["cooldown_seconds"]:
                return None

        self.last_capture_time[region_key] = now
        self.violation_count += 1

        # Draw bounding box on copy
        annotated = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)

        label = f"NO HELMET ({confidence:.0%})"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Add timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, f"Captured: {ts}", (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Save file
        filename = f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        filepath = str(self.capture_dir / filename)
        cv2.imwrite(filepath, annotated)

        self.logger.log(filepath, confidence)
        print(f"[CAPTURE] Violation #{self.violation_count} saved → {filepath}")
        return filepath

    def annotate_frame(self, frame, violators):
        """Draw detections on frame and return annotated copy."""
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # HUD overlay
        cv2.rectangle(annotated, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.putText(annotated, "HELMET DETECTION SYSTEM", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(annotated, f"Violations Today: {self.logger.total()}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(annotated, ts, (w - 250, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Draw violators
        for v in violators:
            x1, y1, x2, y2 = v["bbox"]
            conf = v["confidence"]

            # Red box for violator
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(annotated, (x1, y1 - 35), (x2, y1), (0, 0, 255), -1)
            cv2.putText(annotated, f"NO HELMET {conf:.0%}", (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Warning flash
            cv2.rectangle(annotated, (0, h - 50), (w, h), (0, 0, 180), -1)
            cv2.putText(annotated, "⚠ VIOLATION DETECTED — CAPTURING PHOTO",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return annotated


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────
def run_detection():
    detector = HelmetDetector(CONFIG)

    # Open video source
    source = CONFIG["video_source"] if CONFIG["video_source"] else CONFIG["camera_index"]
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video source: {source}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])

    print("[INFO] Helmet Detection System Running...")
    print("[INFO] Press 'q' to quit, 's' to manually capture frame\n")

    frame_count = 0
    process_every_n = 3  # Process every 3rd frame for performance

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video or camera disconnected.")
            break

        frame_count += 1

        # Run detection every N frames
        if frame_count % process_every_n == 0:
            violators = detector.detect(frame)

            # Capture photos of violators
            for v in violators:
                detector.capture_violation(frame, v["bbox"], v["confidence"])

            # Annotate and show
            display_frame = detector.annotate_frame(frame, violators)
        else:
            display_frame = frame

        cv2.imshow("Helmet Detection System", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Manual capture
            path = f"captures/manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(path, frame)
            print(f"[MANUAL] Frame saved → {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[DONE] Total violations logged: {detector.logger.total()}")
    print(f"[DONE] Photos saved in: {CONFIG['capture_dir']}/")


if __name__ == "__main__":
    run_detection()
