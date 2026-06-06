"""
detect.py
─────────
Real-time helmet violation detection using a trained YOLOv8 model.

Features:
  • Webcam / video file / image input
  • Bounding boxes: GREEN = helmet ✅ | RED = no_helmet ❌
  • Auto-captures and saves violation images to violations/ folder
  • On-screen HUD: FPS, violation count, detection stats
  • Press 'q' to quit, 's' to save screenshot manually

Usage:
    python detect.py                                        # webcam
    python detect.py --source video.mp4                    # video file
    python detect.py --source image.jpg                    # single image
    python detect.py --weights runs/detect/helmet_train/weights/best.pt
    python detect.py --source 0 --conf 0.5 --show-vid
"""

import argparse
import time
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


# ── Detection Configuration ───────────────────────────────────────────────────
DEFAULT_WEIGHTS    = "runs/detect/helmet_train/weights/best.pt"
FALLBACK_WEIGHTS   = "yolov8s.pt"          # Uses COCO if trained weights missing
CONF_THRESHOLD     = 0.45                  # Minimum confidence to show detection
IOU_THRESHOLD      = 0.45                  # NMS IoU threshold
IMG_SIZE           = 640

VIOLATIONS_DIR     = Path("violations")

# Class names must match data.yaml
CLASS_NAMES = {0: "helmet", 1: "no_helmet"}

# Colors (BGR)
COLOR_HELMET    = (34, 197, 94)    # Green
COLOR_VIOLATION = (30, 30, 220)    # Red
COLOR_HUD_BG    = (20, 20, 20)
COLOR_HUD_TEXT  = (255, 255, 255)
COLOR_WARN_TEXT = (30, 30, 220)

# Violation saving: cooldown seconds between saves per detection region
VIOLATION_COOLDOWN = 3.0
# ──────────────────────────────────────────────────────────────────────────────


class HelmetDetector:
    def __init__(self, weights_path: str, conf: float, iou: float):
        try:
            from ultralytics import YOLO
        except ImportError:
            print("❌ ultralytics not installed.  Run: pip install ultralytics")
            sys.exit(1)

        print(f"🔄 Loading model: {weights_path}")
        self.model = YOLO(weights_path)
        self.conf  = conf
        self.iou   = iou

        VIOLATIONS_DIR.mkdir(exist_ok=True)

        self.total_violations  = 0
        self.last_save_time    = 0.0
        self.frame_times: list = []

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, frame: np.ndarray):
        """Run YOLOv8 inference on a single BGR frame."""
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=IMG_SIZE,
            verbose=False,
            stream=False,
        )
        return results[0]

    # ── Drawing ───────────────────────────────────────────────────────────────
    def draw_boxes(self, frame: np.ndarray, result) -> tuple[np.ndarray, int]:
        """Draw bounding boxes and return (annotated_frame, violation_count)."""
        violation_count = 0

        if result.boxes is None or len(result.boxes) == 0:
            return frame, 0

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label  = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            is_violation = (cls_id == 1)   # no_helmet
            color = COLOR_VIOLATION if is_violation else COLOR_HELMET
            if is_violation:
                violation_count += 1

            # Box
            thickness = 3 if is_violation else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label background
            text      = f"{'⚠ ' if is_violation else ''}{label} {conf:.0%}"
            font      = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, 2)
            label_y   = max(y1 - 5, th + 5)
            cv2.rectangle(frame,
                          (x1, label_y - th - 4),
                          (x1 + tw + 4, label_y + baseline),
                          color, cv2.FILLED)
            cv2.putText(frame, text,
                        (x1 + 2, label_y),
                        font, font_scale, (255, 255, 255), 2, cv2.LINE_AA)

            # Flashing red tint on violation boxes
            if is_violation:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), COLOR_VIOLATION, cv2.FILLED)
                cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)

        return frame, violation_count

    def draw_hud(self, frame: np.ndarray, fps: float,
                 violation_count: int, total: int) -> np.ndarray:
        """Draw heads-up display overlay."""
        h, w = frame.shape[:2]

        # Semi-transparent HUD bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), COLOR_HUD_BG, cv2.FILLED)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        font  = cv2.FONT_HERSHEY_SIMPLEX
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 28),
                    font, 0.65, COLOR_HUD_TEXT, 2, cv2.LINE_AA)
        # Active violations
        color = COLOR_WARN_TEXT if violation_count > 0 else COLOR_HELMET
        label = f"VIOLATIONS NOW: {violation_count}"
        cv2.putText(frame, label, (160, 28),
                    font, 0.65, color, 2, cv2.LINE_AA)
        # Cumulative saves
        cv2.putText(frame, f"TOTAL SAVED: {total}", (420, 28),
                    font, 0.65, COLOR_HUD_TEXT, 2, cv2.LINE_AA)
        # Live indicator
        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 120, 28),
                    font, 0.55, (120, 120, 120), 1, cv2.LINE_AA)

        return frame

    # ── Violation Saving ──────────────────────────────────────────────────────
    def maybe_save_violation(self, frame: np.ndarray, violation_count: int):
        """Save frame if a violation is detected and cooldown has elapsed."""
        if violation_count == 0:
            return
        now = time.time()
        if now - self.last_save_time < VIOLATION_COOLDOWN:
            return
        self.last_save_time = now
        self.total_violations += 1
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = VIOLATIONS_DIR / f"violation_{ts}_{violation_count}person.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"🚨 Violation saved: {filename}")

    def save_screenshot(self, frame: np.ndarray, tag: str = "manual"):
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = VIOLATIONS_DIR / f"screenshot_{tag}_{ts}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"📸 Screenshot saved: {filename}")

    # ── FPS tracking ──────────────────────────────────────────────────────────
    def update_fps(self, t_start: float) -> float:
        elapsed = time.time() - t_start
        self.frame_times.append(elapsed)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        avg = sum(self.frame_times) / len(self.frame_times)
        return 1.0 / avg if avg > 0 else 0.0

    # ── Main Loop ─────────────────────────────────────────────────────────────
    def run_video(self, source, show_window: bool = True):
        """Main detection loop for webcam or video file."""
        # Convert numeric string to int for webcam index
        if str(source).isdigit():
            source = int(source)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Cannot open source: {source}")
            sys.exit(1)

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
        print(f"📷 Source: {source}  |  {width}×{height}  |  {fps_in:.1f} FPS")
        print("Controls: [q] quit   [s] save screenshot\n")

        fps = 0.0
        while True:
            t_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("⏹  Stream ended or frame read failed.")
                break

            # Detect
            result = self.predict(frame)
            frame, vcount = self.draw_boxes(frame, result)
            self.maybe_save_violation(frame, vcount)
            fps = self.update_fps(t_start)
            frame = self.draw_hud(frame, fps, vcount, self.total_violations)

            if show_window:
                cv2.imshow("🪖 Helmet Detection System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("👋 Quit by user.")
                    break
                elif key == ord("s"):
                    self.save_screenshot(frame)

        cap.release()
        if show_window:
            cv2.destroyAllWindows()

        print(f"\n📊 Session summary:")
        print(f"   Total violations captured : {self.total_violations}")
        print(f"   Violation images saved to : {VIOLATIONS_DIR}/")

    def run_image(self, image_path: str):
        """Detect on a single image and save annotated result."""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"❌ Cannot read image: {image_path}")
            sys.exit(1)

        result = self.predict(frame)
        frame, vcount = self.draw_boxes(frame, result)
        frame = self.draw_hud(frame, 0, vcount, vcount)

        out_path = VIOLATIONS_DIR / f"result_{Path(image_path).stem}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"✅ Result saved: {out_path}")

        cv2.imshow("Helmet Detection — Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Helmet Violation Detector")
    parser.add_argument("--weights", default=None,
                        help="Path to trained .pt weights file")
    parser.add_argument("--source",  default="0",
                        help="Source: 0/1 (webcam), video.mp4, image.jpg")
    parser.add_argument("--conf",    type=float, default=CONF_THRESHOLD,
                        help="Confidence threshold (default 0.45)")
    parser.add_argument("--iou",     type=float, default=IOU_THRESHOLD,
                        help="NMS IoU threshold (default 0.45)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't display window (save only)")
    return parser.parse_args()


def resolve_weights(user_path):
    """Pick the best available weights."""
    if user_path and Path(user_path).exists():
        return user_path
    best = Path(DEFAULT_WEIGHTS)
    if best.exists():
        print(f"ℹ️  Using trained weights: {best}")
        return str(best)
    print(f"⚠️  Trained weights not found at {DEFAULT_WEIGHTS}")
    print(f"   Using fallback COCO weights: {FALLBACK_WEIGHTS}")
    print("   (Run train.py first for helmet-specific detection)\n")
    return FALLBACK_WEIGHTS


def main():
    print("\n" + "=" * 55)
    print("     🪖  Helmet Violation Detection System")
    print("=" * 55 + "\n")

    args    = parse_args()
    weights = resolve_weights(args.weights)

    detector = HelmetDetector(weights, args.conf, args.iou)

    source = args.source
    # Detect input type
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if Path(source).suffix.lower() in image_exts and Path(source).exists():
        detector.run_image(source)
    else:
        detector.run_video(source, show_window=not args.no_show)


if __name__ == "__main__":
    main()