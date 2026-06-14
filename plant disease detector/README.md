# 🌿 Plant Disease Detection App

Upload a photo of a plant leaf and instantly detect diseases like blight, rust, or mildew using YOLOv8 fine-tuned on the PlantVillage dataset.

## 🚀 Live Demo
Deployed on HuggingFace Spaces — try it without any setup!

## Features
- Detects 10+ plant diseases across tomato, potato, corn, and more
- Confidence score per disease class
- Treatment recommendation per detected disease
- Deployed on HuggingFace Spaces (free hosting)
- Gradio web UI — no coding needed to try it

## Dataset
PlantVillage dataset (54,000+ images, 38 classes):
https://universe.roboflow.com/brad-dwyer/plantdoc

## Setup
```bash
pip install -r requirements.txt
python train.py       # fine-tune (optional)
python app.py         # launch Gradio UI locally
```

## Deploy to HuggingFace Spaces
```bash
pip install huggingface_hub
python deploy.py
```

## Tech Stack
- YOLOv8 (Ultralytics)
- Gradio
- HuggingFace Spaces
- Roboflow (dataset)
