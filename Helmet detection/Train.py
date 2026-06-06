"""
train.py
────────
Train a YOLOv8 model on your helmet / no-helmet dataset.

Usage:
    python train.py                        # default settings
    python train.py --epochs 100           # custom epochs
    python train.py --model yolov8m.pt     # larger model
    python train.py --resume               # resume interrupted training
"""

import argparse
import os
import sys
from pathlib import Path


# ── Training Configuration ────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Model size: yolov8n (nano), yolov8s (small), yolov8m (medium), yolov8l (large)
    # Start with 'n' or 's' for fast iteration; use 'm'/'l' for best accuracy.
    "model"     : "yolov8s.pt",

    "data"      : "data.yaml",       # Dataset config path
    "epochs"    : 50,                # Training epochs (50–150 recommended)
    "imgsz"     : 640,               # Input image size (pixels)
    "batch"     : 16,                # Batch size (-1 = auto)
    "workers"   : 4,                 # DataLoader workers
    "device"    : "",                # "" = auto-detect GPU/CPU, "cpu" = force CPU
    "project"   : "runs/detect",     # Output folder
    "name"      : "helmet_train",    # Run name
    "exist_ok"  : True,              # Overwrite existing run
    "pretrained": True,              # Use COCO pretrained weights
    "optimizer" : "AdamW",           # SGD | Adam | AdamW
    "lr0"       : 0.001,             # Initial learning rate
    "patience"  : 15,                # Early stopping patience (epochs)
    "save_period": 10,               # Save checkpoint every N epochs
    "val"       : True,              # Validate during training
    "plots"     : True,              # Save training plots
    "verbose"   : True,

    # Augmentation (helps with small datasets)
    "hsv_h"     : 0.015,             # Hue augmentation
    "hsv_s"     : 0.7,               # Saturation augmentation
    "hsv_v"     : 0.4,               # Value augmentation
    "degrees"   : 5.0,               # Rotation ±degrees
    "translate" : 0.1,               # Translation ±fraction
    "scale"     : 0.5,               # Scale ±gain
    "fliplr"    : 0.5,               # Horizontal flip probability
    "mosaic"    : 1.0,               # Mosaic augmentation (very helpful)
    "mixup"     : 0.1,               # MixUp augmentation
    "copy_paste": 0.0,               # Copy-paste augmentation
}
# ──────────────────────────────────────────────────────────────────────────────


def check_dataset():
    """Verify dataset structure before training."""
    required = [
        Path("dataset/images/train"),
        Path("dataset/images/val"),
        Path("dataset/labels/train"),
        Path("dataset/labels/val"),
        Path("data.yaml"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("❌ Missing required paths:")
        for m in missing:
            print(f"   {m}")
        print("\nRun first:  python prepare_dataset.py")
        sys.exit(1)

    n_train = len(list(Path("dataset/images/train").glob("*.jpg")))
    n_val   = len(list(Path("dataset/images/val").glob("*.jpg")))

    if n_train == 0:
        print("❌ No training images found in dataset/images/train/")
        sys.exit(1)

    print(f"✅ Dataset OK  — Train: {n_train} images | Val: {n_val} images")
    return n_train, n_val


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Helmet Detector")
    parser.add_argument("--model",   default=DEFAULT_CONFIG["model"],
                        help="YOLOv8 model variant (yolov8n/s/m/l/x.pt)")
    parser.add_argument("--epochs",  type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--imgsz",   type=int, default=DEFAULT_CONFIG["imgsz"])
    parser.add_argument("--batch",   type=int, default=DEFAULT_CONFIG["batch"])
    parser.add_argument("--device",  default=DEFAULT_CONFIG["device"],
                        help="Device: '' (auto), '0' (GPU 0), 'cpu'")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint")
    parser.add_argument("--name",    default=DEFAULT_CONFIG["name"])
    return parser.parse_args()


def print_header():
    print("\n" + "=" * 60)
    print("     🪖  Helmet Detection — YOLOv8 Training")
    print("=" * 60)


def main():
    print_header()

    args = parse_args()

    # Lazy import so error messages are clean
    try:
        from ultralytics import YOLO
        import torch
        print(f"✅ PyTorch  {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run:  pip install ultralytics")
        sys.exit(1)

    check_dataset()

    # Build training kwargs
    train_kwargs = {**DEFAULT_CONFIG}
    train_kwargs.update({
        "model"  : args.model,
        "epochs" : args.epochs,
        "imgsz"  : args.imgsz,
        "batch"  : args.batch,
        "device" : args.device,
        "name"   : args.name,
        "resume" : args.resume,
    })
    # Remove 'model' from kwargs (passed separately)
    model_path = train_kwargs.pop("model")

    print(f"\n🚀 Starting training:")
    print(f"   Model   : {model_path}")
    print(f"   Epochs  : {train_kwargs['epochs']}")
    print(f"   ImgSize : {train_kwargs['imgsz']}")
    print(f"   Batch   : {train_kwargs['batch']}")
    print(f"   Device  : {train_kwargs['device'] or 'auto'}")
    print(f"   Output  : {train_kwargs['project']}/{train_kwargs['name']}/")
    print()

    # Load and train
    model = YOLO(model_path)
    results = model.train(**train_kwargs)

    # Post-training summary
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    best_weights = Path(train_kwargs["project"]) / train_kwargs["name"] / "weights" / "best.pt"
    if best_weights.exists():
        print(f"   Best weights : {best_weights}")
        print(f"\n🎯 To run detection:")
        print(f"   python detect.py --weights {best_weights}")
    print("=" * 60 + "\n")

    # Quick validation on best weights
    print("📊 Running validation with best weights...")
    best_model = YOLO(str(best_weights))
    val_results = best_model.val(data="data.yaml", imgsz=train_kwargs["imgsz"])
    print(f"\n   mAP50      : {val_results.box.map50:.4f}")
    print(f"   mAP50-95   : {val_results.box.map:.4f}")

    return results


if __name__ == "__main__":
    main()