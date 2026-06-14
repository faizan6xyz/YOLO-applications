# Treatment recommendations per disease class
TREATMENTS = {
    "Tomato_Late_blight": "Apply copper-based fungicide. Remove infected leaves. Avoid overhead watering.",
    "Tomato_Early_blight": "Use chlorothalonil fungicide. Rotate crops annually. Ensure good air circulation.",
    "Potato_Late_blight": "Apply mancozeb or metalaxyl fungicide. Destroy infected plants.",
    "Corn_Common_rust": "Apply propiconazole fungicide. Plant resistant varieties.",
    "Tomato_Leaf_Mold": "Improve ventilation. Apply fungicide with chlorothalonil.",
    "Healthy": "Plant looks healthy! Continue regular watering and fertilizing.",
}

def get_treatment(disease_label: str) -> str:
    for key in TREATMENTS:
        if key.lower() in disease_label.lower():
            return TREATMENTS[key]
    return "Consult a local agronomist for treatment advice."
