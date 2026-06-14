"""
Fine-tune YOLOv8 on the PPE dataset from Roboflow.
Set your Roboflow API key in environment: export ROBOFLOW_API_KEY=your_key
"""
from roboflow import Roboflow
from ultralytics import YOLO
import os

RF_API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")
PROJECT = "construction-site-safety"
VERSION = 1

def download_dataset():
    rf = Roboflow(api_key=RF_API_KEY)
    project = rf.workspace("roboflow-universe-projects").project(PROJECT)
    dataset = project.version(VERSION).download("yolov8")
    return dataset.location

def train(data_path):
    model = YOLO("yolov8n.pt")
    model.train(
        data=f"{data_path}/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        name="ppe_detector",
        project="runs/train",
        patience=10,
    )
    print("Training complete. Best weights: runs/train/ppe_detector/weights/best.pt")

if __name__ == "__main__":
    data_path = download_dataset()
    train(data_path)
