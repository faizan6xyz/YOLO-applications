from ultralytics import YOLO

model = YOLO("best.pt")

# any video you want to predict just place it in "Heelmet new/predict/video/" folder


results = model.predict(
    source="Heelmet new/predict/video/",
    save=True
)

for result in results[:5]:
    result.show()