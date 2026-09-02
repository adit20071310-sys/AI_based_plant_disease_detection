# ============================================
# PLANT DISEASE DETECTOR — Flask Backend
# Transfer Learning (MobileNetV2) version
# ============================================

import os
import json
import numpy as np

from flask import Flask, render_template, request
from PIL import Image
import tensorflow as tf

# ============================================
# FLASK APP
# ============================================

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ============================================
# LOAD MODEL + CLASS NAMES
# ============================================

model = tf.keras.models.load_model("model_cnn.h5")

with open("class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

IMG_SIZE = 224      # Must match MobileNetV2 training size

# ============================================
# CONFIDENCE THRESHOLD
# If top prediction is below this, show
# "uncertain" warning to the user
# ============================================

CONFIDENCE_THRESHOLD = 40.0   # percent

# ============================================
# TREATMENTS MAP
# ============================================

TREATMENTS = {
    "Potato___Early_blight": {
        "icon": "🥔", "plant": "Potato", "disease": "Early Blight",
        "severity": "Moderate",
        "treatment": "Apply copper-based fungicide every 7–10 days. Remove and destroy infected lower leaves. Avoid overhead irrigation. Ensure good airflow around plants.",
        "prevention": "Rotate crops yearly. Use certified disease-free seed potatoes. Mulch around plants to prevent soil splash.",
    },
    "Potato___Late_blight": {
        "icon": "🥔", "plant": "Potato", "disease": "Late Blight",
        "severity": "Severe",
        "treatment": "Apply systemic fungicide (Mancozeb or Ridomil) immediately. Remove all infected plant material and bag it. Avoid wetting foliage when watering.",
        "prevention": "Plant resistant varieties. Monitor humidity levels closely. Destroy infected crops to prevent spread to neighboring plants.",
    },
    "Potato___healthy": {
        "icon": "🥔", "plant": "Potato", "disease": "Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Plant looks great!",
        "prevention": "Maintain regular watering and balanced fertilization schedule.",
    },
    "Tomato___Early_blight": {
        "icon": "🍅", "plant": "Tomato", "disease": "Early Blight",
        "severity": "Moderate",
        "treatment": "Use chlorothalonil or copper fungicide. Prune lower infected leaves and improve air circulation between plants.",
        "prevention": "Mulch soil around plants. Stake or cage plants. Water at the base only, never overhead.",
    },
    "Tomato___Late_blight": {
        "icon": "🍅", "plant": "Tomato", "disease": "Late Blight",
        "severity": "Severe",
        "treatment": "Apply fungicide immediately (Mancozeb or Ridomil Gold). Remove and destroy heavily affected plants to stop spread.",
        "prevention": "Plant resistant varieties. Avoid dense planting. Inspect plants every 2–3 days during humid weather.",
    },
    "Tomato___healthy": {
        "icon": "🍅", "plant": "Tomato", "disease": "Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Plant is in great condition.",
        "prevention": "Continue regular feeding and watering routine. Inspect weekly.",
    },
    "Pepper__bell___Bacterial_spot": {
        "icon": "🫑", "plant": "Pepper", "disease": "Bacterial Spot",
        "severity": "Moderate",
        "treatment": "Apply copper-based bactericide. Remove and bag infected leaves. Avoid working with plants when they are wet.",
        "prevention": "Use disease-free transplants. Avoid overhead watering. Disinfect tools between plants.",
    },
    "Pepper__bell___healthy": {
        "icon": "🫑", "plant": "Pepper", "disease": "Healthy",
        "severity": "None",
        "treatment": "No treatment needed. Plant is healthy.",
        "prevention": "Maintain proper plant spacing and balanced nutrition.",
    },
}

SEVERITY_COLORS = {
    "None": "healthy", "Moderate": "warning", "Severe": "danger"
}

# ============================================
# IMAGE PREPARATION
# ============================================

def prepare_image(image_path):
    """
    Load image with PIL, resize to 224x224,
    normalize to [0,1], return batch tensor.
    Using PIL instead of OpenCV avoids BGR/RGB
    confusion which is a common cause of bad
    predictions.
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)       # shape: (1, 224, 224, 3)
    return arr

# ============================================
# HOME PAGE
# ============================================

@app.route('/')
def home():
    return render_template("index.html")

# ============================================
# PREDICTION ROUTE
# ============================================

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template("index.html", error="No image uploaded.")

    file = request.files['image']
    if file.filename == '':
        return render_template("index.html", error="No file selected.")

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Prepare image
    processed = prepare_image(filepath)

    # CNN prediction
    predictions = model.predict(processed)[0]

    # Top 3 results
    top_indices = np.argsort(predictions)[::-1][:3]

    results  = []
    low_conf = False

    for i in top_indices:
        class_key  = CLASS_NAMES[i]
        confidence = round(float(predictions[i]) * 100, 2)

        if i == top_indices[0] and confidence < CONFIDENCE_THRESHOLD:
            low_conf = True

        info = TREATMENTS.get(class_key, {
            "icon": "🌿",
            "plant": class_key.split("___")[0],
            "disease": class_key.split("___")[-1].replace("_", " "),
            "severity": "Unknown",
            "treatment": "Consult an agricultural expert.",
            "prevention": "Monitor plant regularly.",
        })

        results.append({
            "class_key": class_key,
            "confidence": confidence,
            "severity_class": SEVERITY_COLORS.get(info.get("severity", "Unknown"), "warning"),
            **info,
        })

    return render_template(
        "result.html",
        results=results,
        filename=file.filename,
        low_conf=low_conf
    )

# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    app.run(debug=True)
