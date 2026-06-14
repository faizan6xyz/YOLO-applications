import gradio as gr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from treatments import get_treatment

MODEL_PATH = "runs/train/plant_disease/weights/best.pt"  # or "yolov8n.pt" for demo

model = YOLO(MODEL_PATH)

def predict(image: Image.Image):
    img_np = np.array(image)
    results = model(img_np, verbose=False)[0]

    detections = []
    for box in results.boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)
        treatment = get_treatment(label)
        detections.append(f"**{label}** ({conf*100:.1f}% confidence)\n→ {treatment}")

    if not detections:
        return "No disease detected. Plant appears healthy! 🌱"

    return "\n\n".join(detections)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload leaf image"),
    outputs=gr.Markdown(label="Detection Results"),
    title="🌿 Plant Disease Detector",
    description="Upload a photo of a plant leaf to detect diseases and get treatment advice.",
    examples=[],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
