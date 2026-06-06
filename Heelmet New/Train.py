from ultralytics import YOLO
if __name__ == "__main__":
    # Load pretrained model
    model = YOLO("yolo11n.pt")
    # Train
    results = model.train(
        data="Heelmet new/dataset/data.yaml",
        epochs=50,          # Number of training epochs
        imgsz=640,
        batch=16,
        device=0,          # GPU (use "cpu" if no GPU)
        workers=4,         # Number of dataloader workers
        patience=20,       # Early stopping (Stop if no improvement)
        project="runs",     # Output folder
        name="helmet_detection",
        save=True,          # Save weights
        verbose=True        # Show logs
    )
    # Validate model
    metrics = model.val()
    # Export model (optional)
    model.export(format="onnx")