from ultralytics import YOLO

model = YOLO("Heelmet new/runs/detect/train-3/weights/best.pt")

# anu image you want to predict just place it in "Heelmet new/predict/images/" folder

results = model.predict(
    source="Heelmet new/predict/images/",
    conf=0.5
)

for result in results[:10]:
    result.show()