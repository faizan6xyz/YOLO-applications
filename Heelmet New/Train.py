from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolo11n.pt")

    model.train(
        data="Heelmet new/dataset/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16
    )