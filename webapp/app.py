import os
import sys
import uuid

import torch
from flask import Flask, jsonify, render_template, request, url_for
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.model import NIH_CLASSES, load_trained_model
from common.inference import predict_tta, preprocess, probs_to_report
from common.gradcam import explain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "..", "models", "FINAL_BEST_EPOCH5_AUC_0.76148.pth")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

_model = None


def get_model():
    """Lazy-load the model once, on first request."""
    global _model
    if _model is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                f"No weights found at {WEIGHTS_PATH}. Train the model on Kaggle "
                f"(see kaggle_train/train_notebook.ipynb), download best_model.pth, "
                f"and place it at models/ or webapp/weights/."
            )
        _model = load_trained_model(WEIGHTS_PATH, device=DEVICE)
    return _model


@app.route("/")
def index():
    weights_ready = os.path.exists(WEIGHTS_PATH)
    return render_template("index.html", weights_ready=weights_ready)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        model = get_model()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    uid = uuid.uuid4().hex[:10]
    orig_path = os.path.join(UPLOAD_DIR, f"{uid}.png")
    pil_image = Image.open(file.stream).convert("RGB")
    pil_image.save(orig_path)

    # 1. TTA prediction (averaged over multiple augmented views)
    mean_probs, top_idx = predict_tta(model, pil_image, device=DEVICE)
    report = probs_to_report(mean_probs)

    # 2. Grad-CAM + arrow explanation for the top predicted class
    input_tensor = preprocess(pil_image).to(DEVICE)
    input_tensor.requires_grad_(True)
    model.zero_grad()
    _, _, overlay_img, arrow_img, hotspot = explain(model, input_tensor, top_idx, pil_image)

    overlay_path = os.path.join(OUTPUT_DIR, f"{uid}_heatmap.png")
    arrow_path = os.path.join(OUTPUT_DIR, f"{uid}_arrow.png")
    overlay_img.save(overlay_path)
    arrow_img.save(arrow_path)

    return jsonify(
        {
            "top_prediction": NIH_CLASSES[top_idx],
            "top_probability": float(mean_probs[top_idx]),
            "report": report,
            "heatmap_url": url_for("static_output", filename=f"{uid}_heatmap.png"),
            "arrow_url": url_for("static_output", filename=f"{uid}_arrow.png"),
            "original_url": url_for("static_upload", filename=f"{uid}.png"),
        }
    )


@app.route("/uploads/<path:filename>")
def static_upload(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/outputs/<path:filename>")
def static_output(filename):
    from flask import send_from_directory
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)