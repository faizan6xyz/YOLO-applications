"""
Train a Custom Helmet Detection Model with YOLOv8
Uses a helmet dataset from Roboflow Universe
"""

from ultralytics import YOLO

# ─────────────────────────────────────────────
# Option 1: Train with your own dataset
# ─────────────────────────────────────────────
# Dataset YAML format (data.yaml):
#
# path: ./dataset
# train: images/train
# val: images/val
# test: images/test
#
# nc: 2
# names: ['helmet', 'no_helmet']

def train_custom_model():
    model = YOLO("yolov8n.pt")  # Start from pretrained weights

    results = model.train(
        data="data.yaml",         # Your dataset config
        epochs=50,
        imgsz=640,
        batch=16,
        name="helmet_detector",
        project="runs/train",
        patience=10,              # Early stopping
        augment=True,
        workers=4,
    )

    print(f"Model saved to: runs/train/helmet_detector/weights/best.pt")
    return results


# ─────────────────────────────────────────────
# Option 2: Use a pre-trained helmet model
# ─────────────────────────────────────────────
# Roboflow has pre-trained helmet models. Use their API:
#
# pip install roboflow
#
# from roboflow import Roboflow
# rf = Roboflow(api_key="YOUR_API_KEY")
# project = rf.workspace("roboflow-100").project("hard-hat-sample")
# dataset = project.version(2).download("yolov8")
#
# Then run train_custom_model() with the downloaded data.yaml


# ─────────────────────────────────────────────
# Validate the trained model
# ─────────────────────────────────────────────
def validate_model(model_path="runs/train/helmet_detector/weights/best.pt"):
    model = YOLO(model_path)
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f}")
    print(f"mAP50-95: {metrics.box.map:.3f}")
    return metrics


# ─────────────────────────────────────────────
# Test on a single image
# ─────────────────────────────────────────────
def test_image(image_path, model_path="runs/train/helmet_detector/weights/best.pt"):
    model = YOLO(model_path)
    results = model(image_path, conf=0.5)
    results[0].show()
    results[0].save(filename="test_result.jpg")


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Train custom model")
    print("2. Validate model")
    print("3. Test on image")

    choice = input("Enter choice (1/2/3): ").strip()

    if choice == "1":
        train_custom_model()
    elif choice == "2":
        validate_model()
    elif choice == "3":
        img = input("Image path: ").strip()
        test_image(img)
