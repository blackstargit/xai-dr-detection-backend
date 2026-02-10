import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from flask_cors import CORS

from scripts.nlp import DiabeticRetinopathyExplainer  # Your NLP explainer module
from scripts.xai import make_gradcam_heatmap  # Your XAI module

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- Configuration ---
image_size = (224, 224)
num_classes = 5
model_path = "./assets/best_resnet50_model.keras"  # Update with your model path
last_conv_layer_name = "conv5_block3_out"  # Update with your layer name

# --- Load Model ---
model = load_model(model_path)

# --- Instantiate Explainer ---
explainer = DiabeticRetinopathyExplainer(threshold=0.7)


# --- Preprocessing Function ---
def preprocess_image_from_bytes(image_bytes, target_size):
    """Preprocesses an image from bytes for the model."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array


def display_gradcam(img, heatmap, predicted_class, explainer, alpha=0.4):
    # Ensure the heatmap is in the range [0, 1]
    if heatmap.max() > 1:
        heatmap = heatmap / 255.0

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.colormaps.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    return superimposed_img


# --- API Routes ---


@app.route("/predict", methods=["POST"])
def predict():
    """Predicts the class of an image and provides an explanation."""
    try:
        # --- Get Image from Request ---
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        image_file = request.files["image"]

        image_bytes = image_file.read()

        # Preprocess the image
        preprocessed_image = preprocess_image_from_bytes(image_bytes, image_size)

        # --- Make Prediction ---
        predictions = model.predict(preprocessed_image)
        predicted_class = np.argmax(predictions[0])

        # --- Generate Grad-CAM Heatmap ---
        heatmap = make_gradcam_heatmap(
            preprocessed_image, model, last_conv_layer_name, pred_index=predicted_class
        )

        # --- Generate Explanation ---
        activated_regions = explainer.analyze_heatmap_regions(heatmap)
        explanation = explainer.generate_explanation(activated_regions)

        # --- Prepare Grad-CAM Image for Response ---
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.resize(image_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)

        gradcam_img = display_gradcam(img_array, heatmap, predicted_class, explainer)

        # Save the Grad-CAM image to a bytes buffer
        buffer = io.BytesIO()
        gradcam_img.save(buffer, format="JPEG")
        gradcam_image_bytes = buffer.getvalue()

        # Encode the Grad-CAM image to base64
        gradcam_image_base64 = base64.b64encode(gradcam_image_bytes).decode("utf-8")

        # --- Return Response ---
        return (
            jsonify(
                {
                    "predicted_class": int(predicted_class),
                    "gradcam_image": gradcam_image_base64,
                    "explanation": explanation,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
