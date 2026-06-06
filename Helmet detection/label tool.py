"""
label_tool.py
─────────────
Interactive annotation tool to create YOLO .txt label files from .jpg images.
Useful if you need to manually label your own images.

Controls:
  Left-click + drag  → Draw bounding box
  H key              → Set class to 'helmet' (0) before drawing
  N key              → Set class to 'no_helmet' (1) before drawing
  Z / Ctrl+Z         → Undo last annotation
  S key              → Save labels to .txt and move to next image
  D key              → Skip image (no annotations)
  Q key              → Quit

Usage:
    python label_tool.py --images dataset/raw/images/
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


CLASS_NAMES = {0: "helmet", 1: "no_helmet"}
CLASS_COLORS = {0: (34, 197, 94), 1: (30, 30, 220)}

LABELS_OUTPUT_DIR = Path("dataset/raw/labels")


class Annotator:
    def __init__(self, images_dir: str):
        self.images_dir  = Path(images_dir)
        self.labels_dir  = LABELS_OUTPUT_DIR
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg"))
            + list(self.images_dir.glob("*.jpeg"))
            + list(self.images_dir.glob("*.JPG"))
        )
        if not self.image_files:
            print(f"❌ No .jpg images found in {images_dir}")
            sys.exit(1)

        print(f"📸 Found {len(self.image_files)} images to annotate.")

        self.current_class = 1   # default: no_helmet (violation)
        self.annotations: list  = []  # [(class_id, x1, y1, x2, y2)]
        self.drawing     = False
        self.ix, self.iy = -1, -1
        self.ex, self.ey = -1, -1
        self.frame       = None
        self.display     = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
            self.ex, self.ey = x, y

        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.ex, self.ey = x, y
            self._refresh_display()

        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.ex, self.ey = x, y
            x1, y1 = min(self.ix, self.ex), min(self.iy, self.ey)
            x2, y2 = max(self.ix, self.ex), max(self.iy, self.ey)
            if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                self.annotations.append((self.current_class, x1, y1, x2, y2))
                print(f"   + Added: {CLASS_NAMES[self.current_class]} "
                      f"({x1},{y1}) → ({x2},{y2})")
            self._refresh_display()

    def _refresh_display(self):
        self.display = self.frame.copy()
        h, w = self.display.shape[:2]

        # Draw saved annotations
        for cls, x1, y1, x2, y2 in self.annotations:
            color = CLASS_COLORS[cls]
            cv2.rectangle(self.display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(self.display, CLASS_NAMES[cls],
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Draw current box being drawn
        if self.drawing and self.ix > 0:
            color = CLASS_COLORS[self.current_class]
            cv2.rectangle(self.display,
                          (self.ix, self.iy), (self.ex, self.ey),
                          color, 2)

        # HUD
        cls_text = f"Class: [{self.current_class}] {CLASS_NAMES[self.current_class]}"
        annot_text = f"Boxes: {len(self.annotations)}"
        cv2.putText(self.display, cls_text, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    CLASS_COLORS[self.current_class], 2)
        cv2.putText(self.display, annot_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(self.display, "H=helmet  N=no_helmet  S=save  Z=undo  D=skip  Q=quit",
                    (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    def to_yolo(self, x1, y1, x2, y2, img_w, img_h):
        """Convert pixel coords to YOLO normalized format."""
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        return cx, cy, bw, bh

    def save_labels(self, img_path: Path):
        if not self.annotations:
            print(f"   ℹ️  No annotations — skipping label file.")
            return
        label_path = self.labels_dir / (img_path.stem + ".txt")
        h, w = self.frame.shape[:2]
        with open(label_path, "w") as f:
            for cls, x1, y1, x2, y2 in self.annotations:
                cx, cy, bw, bh = self.to_yolo(x1, y1, x2, y2, w, h)
                f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"   💾 Saved: {label_path} ({len(self.annotations)} box(es))")

    def run(self):
        already_labeled = {p.stem for p in self.labels_dir.glob("*.txt")}
        remaining = [p for p in self.image_files if p.stem not in already_labeled]
        print(f"   Already labeled: {len(self.image_files) - len(remaining)}")
        print(f"   To annotate    : {len(remaining)}\n")

        if not remaining:
            print("✅ All images already labeled!")
            return

        cv2.namedWindow("Annotator", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Annotator", 1100, 700)
        cv2.setMouseCallback("Annotator", self.mouse_callback)

        for idx, img_path in enumerate(remaining):
            self.annotations = []
            self.frame = cv2.imread(str(img_path))
            if self.frame is None:
                print(f"   ⚠️  Cannot read {img_path.name}, skipping.")
                continue

            self._refresh_display()
            print(f"\n[{idx+1}/{len(remaining)}] {img_path.name}")

            while True:
                cv2.imshow("Annotator", self.display)
                key = cv2.waitKey(20) & 0xFF

                if key == ord("h"):
                    self.current_class = 0
                    print("   🟢 Class → helmet (0)")
                    self._refresh_display()
                elif key == ord("n"):
                    self.current_class = 1
                    print("   🔴 Class → no_helmet (1)")
                    self._refresh_display()
                elif key == ord("z"):
                    if self.annotations:
                        removed = self.annotations.pop()
                        print(f"   ↩  Undo: {CLASS_NAMES[removed[0]]}")
                        self._refresh_display()
                elif key == ord("s"):
                    self.save_labels(img_path)
                    break
                elif key == ord("d"):
                    print(f"   ⏭  Skipped.")
                    break
                elif key == ord("q"):
                    print("\n👋 Quit annotation tool.")
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()
        print("\n✅ Annotation complete!")
        print(f"   Labels saved to: {self.labels_dir}/")
        print("   Next step: python prepare_dataset.py")


def main():
    parser = argparse.ArgumentParser(description="YOLO Label Annotation Tool")
    parser.add_argument("--images", default="dataset/raw/images",
                        help="Directory containing .jpg images to label")
    args = parser.parse_args()

    annotator = Annotator(args.images)
    annotator.run()


if __name__ == "__main__":
    main()