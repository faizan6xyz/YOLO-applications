from roboflow import Roboflow
from ultralytics import YOLO
import os

RF_API_KEY = os.getenv("ROBOFLOW_API_KEY", "YOUR_API_KEY_HERE")

def main():
    rf = Roboflow(api_key=RF_API_KEY)
    project = rf.workspace("brad-dwyer").project("plantdoc")
    dataset = project.version(1).download("yolov8")

    model = YOLO("yolov8s.pt")
    model.train(
        data=f"{dataset.location}/data.yaml",
        epochs=80,
        imgsz=640,
        batch=16,
        name="plant_disease",
        project="runs/train",
    )

if __name__ == "__main__":
    main()
