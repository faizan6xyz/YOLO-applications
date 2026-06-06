"""
prepare_dataset.py
──────────────────
Automatically splits your raw JPG + TXT label files into
train (80%) and val (20%) sets with the correct YOLO folder structure.

Usage:
    1. Place ALL your .jpg images in:   dataset/raw/images/
    2. Place ALL your .txt labels in:   dataset/raw/labels/
    3. Run:  python prepare_dataset.py
"""

import os
import shutil
import random
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
RAW_IMAGES_DIR = Path("dataset/raw/images")   # Your source .jpg files
RAW_LABELS_DIR = Path("dataset/raw/labels")   # Your source .txt label files
TRAIN_SPLIT    = 0.8                           # 80% training, 20% validation
RANDOM_SEED    = 42
# ──────────────────────────────────────────────────────────────────────────────

def create_dirs():
    """Create the YOLO folder structure."""
    dirs = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/raw/images",
        "dataset/raw/labels",
        "violations",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    print("✅ Directory structure created.")


def validate_pairs(image_files):
    """Check that every image has a matching label file."""
    valid, missing_labels = [], []
    for img_path in image_files:
        label_path = RAW_LABELS_DIR / (img_path.stem + ".txt")
        if label_path.exists():
            valid.append(img_path)
        else:
            missing_labels.append(img_path.name)

    if missing_labels:
        print(f"\n⚠️  {len(missing_labels)} images have NO label file — skipping:")
        for name in missing_labels[:10]:
            print(f"   - {name}")
        if len(missing_labels) > 10:
            print(f"   ... and {len(missing_labels) - 10} more")

    return valid


def split_and_copy(image_files):
    """Shuffle, split, and copy image+label pairs to train/val folders."""
    random.seed(RANDOM_SEED)
    random.shuffle(image_files)

    split_idx  = int(len(image_files) * TRAIN_SPLIT)
    train_imgs = image_files[:split_idx]
    val_imgs   = image_files[split_idx:]

    def copy_pair(img_path, subset):
        dst_img   = Path(f"dataset/images/{subset}") / img_path.name
        src_label = RAW_LABELS_DIR / (img_path.stem + ".txt")
        dst_label = Path(f"dataset/labels/{subset}") / (img_path.stem + ".txt")
        shutil.copy2(img_path, dst_img)
        shutil.copy2(src_label, dst_label)

    print(f"\n📂 Copying {len(train_imgs)} training pairs...")
    for img in train_imgs:
        copy_pair(img, "train")

    print(f"📂 Copying {len(val_imgs)} validation pairs...")
    for img in val_imgs:
        copy_pair(img, "val")

    return len(train_imgs), len(val_imgs)


def verify_labels(subset):
    """Quick sanity check on label file format."""
    label_dir   = Path(f"dataset/labels/{subset}")
    label_files = list(label_dir.glob("*.txt"))
    errors      = []

    for lf in label_files[:50]:   # Check first 50 files
        with open(lf) as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{lf.name} line {line_no}: expected 5 values, got {len(parts)}")
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(p) for p in parts[1:]]
                    if cls_id not in (0, 1):
                        errors.append(f"{lf.name} line {line_no}: unknown class id {cls_id} (use 0=helmet, 1=no_helmet)")
                    if not all(0.0 <= c <= 1.0 for c in coords):
                        errors.append(f"{lf.name} line {line_no}: coordinates must be in [0, 1]")
                except ValueError:
                    errors.append(f"{lf.name} line {line_no}: invalid numeric values")

    return errors


def print_class_stats(subset):
    """Count helmet vs no_helmet annotations."""
    label_dir = Path(f"dataset/labels/{subset}")
    counts    = {0: 0, 1: 0}
    for lf in label_dir.glob("*.txt"):
        with open(lf) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        counts[int(line.split()[0])] += 1
                    except (ValueError, KeyError):
                        pass
    print(f"   Class 0 (helmet)    : {counts[0]:>6} annotations")
    print(f"   Class 1 (no_helmet) : {counts[1]:>6} annotations")


def main():
    print("=" * 55)
    print("  Helmet Detection — Dataset Preparation")
    print("=" * 55)

    create_dirs()

    # Find all JPG images (case-insensitive)
    image_files = (
        list(RAW_IMAGES_DIR.glob("*.jpg"))
        + list(RAW_IMAGES_DIR.glob("*.JPG"))
        + list(RAW_IMAGES_DIR.glob("*.jpeg"))
        + list(RAW_IMAGES_DIR.glob("*.JPEG"))
    )

    if not image_files:
        print(f"\n❌ No .jpg images found in {RAW_IMAGES_DIR}/")
        print("   Please add your images there and re-run.")
        return

    print(f"\n📸 Found {len(image_files)} image(s) in {RAW_IMAGES_DIR}/")

    valid_images = validate_pairs(image_files)
    if not valid_images:
        print("\n❌ No valid image-label pairs found. Aborting.")
        return

    print(f"✅ {len(valid_images)} valid image-label pair(s) ready to split.")

    n_train, n_val = split_and_copy(valid_images)

    # Verify labels
    print("\n🔍 Validating label format...")
    for subset in ("train", "val"):
        errors = verify_labels(subset)
        if errors:
            print(f"\n⚠️  Errors in {subset} labels:")
            for e in errors[:5]:
                print(f"   {e}")
        else:
            print(f"   ✅ {subset} labels: OK")

    # Stats
    print("\n📊 Dataset Summary")
    print(f"   Train : {n_train} images")
    print(f"   Val   : {n_val} images")
    print("\n   Train class distribution:")
    print_class_stats("train")
    print("   Val class distribution:")
    print_class_stats("val")

    print("\n✅ Dataset prepared! Run next:")
    print("   python train.py\n")


if __name__ == "__main__":
    main()