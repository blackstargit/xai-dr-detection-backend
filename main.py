import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
from config import Config
from services.model_service import ModelService
from services.xai_service import XaiService
from services.nlp_service import NlpService
from utils.image_utils import preprocess_image_from_bytes, overlay_heatmap, image_to_base64
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load global services with fallback handling
model_service = None
xai_service = None
nlp_service = NlpService(Config.EXPLANATION_THRESHOLD)
MOCK_MODE = False

try:
    model_service = ModelService(Config.MODEL_PATH)
    xai_service = XaiService(model_service.model, Config.LAST_CONV_LAYER_NAME)
    logger.info("Deep learning model and XAI service initialized successfully.")
except Exception as e:
    logger.warning(f"Failed to load model from {Config.MODEL_PATH}. Starting in MOCK_MODE: {e}")
    MOCK_MODE = True

def generate_mock_heatmap(predicted_class):
    """Generates a 2D Gaussian heatmap centered at class-dependent coordinates."""
    x, y = np.meshgrid(np.linspace(-1, 1, 224), np.linspace(-1, 1, 224))
    heatmap = np.zeros((224, 224))
    
    # Place Gaussian hotspots depending on class
    if predicted_class == 0:  # No DR
        d = np.sqrt(x*x + y*y)
        heatmap = np.exp(-((d - 0)**2 / (2 * 0.1**2))) * 0.15
    elif predicted_class == 1:  # Mild
        d = np.sqrt((x - 0.3)**2 + (y - 0.2)**2)
        heatmap = np.exp(-((d - 0)**2 / (2 * 0.15**2)))
    elif predicted_class == 2:  # Moderate
        d1 = np.sqrt((x + 0.2)**2 + (y + 0.3)**2)
        d2 = np.sqrt((x - 0.3)**2 + (y - 0.2)**2)
        heatmap = np.exp(-((d1 - 0)**2 / (2 * 0.12**2))) + np.exp(-((d2 - 0)**2 / (2 * 0.15**2)))
    elif predicted_class == 3:  # Severe
        d1 = np.sqrt((x - 0.1)**2 + (y - 0.1)**2)
        d2 = np.sqrt((x + 0.4)**2 + (y - 0.4)**2)
        heatmap = np.exp(-((d1 - 0)**2 / (2 * 0.25**2))) + np.exp(-((d2 - 0)**2 / (2 * 0.1**2))) * 0.8
    else:  # Proliferative
        d1 = np.sqrt((x + 0.2)**2 + (y + 0.2)**2)
        d2 = np.sqrt((x - 0.4)**2 + (y + 0.3)**2)
        d3 = np.sqrt((x - 0.1)**2 + (y - 0.5)**2)
        heatmap = np.exp(-((d1 - 0)**2 / (2 * 0.3**2))) + np.exp(-((d2 - 0)**2 / (2 * 0.2**2))) + np.exp(-((d3 - 0)**2 / (2 * 0.15**2)))
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    return heatmap

@app.route("/predict", methods=["POST"])
def predict():
    """Predicts the severity of diabetic retinopathy and returns Grad-CAM visualisations."""
    try:
        if "image" not in request.files:
            logger.warning("Request failed: No image file provided.")
            return jsonify({"error": "No image file provided in request."}), 400
            
        image_file = request.files["image"]
        image_bytes = image_file.read()
        
        # 1. Preprocess
        preprocessed_image, original_pil = preprocess_image_from_bytes(image_bytes, Config.IMAGE_SIZE)
        
        if MOCK_MODE:
            # Generate deterministic predicted class based on image size or random sample
            predicted_class = np.random.choice([0, 1, 2, 3, 4], p=[0.3, 0.25, 0.2, 0.15, 0.1])
            probabilities = np.zeros(5)
            probabilities[predicted_class] = 0.85
            # Distribute remaining 15% randomly
            rem = 0.15 / 4
            for idx in range(5):
                if idx != predicted_class:
                    probabilities[idx] = rem
                    
            heatmap = generate_mock_heatmap(predicted_class)
            logger.info(f"[MOCK] Predicted Class: {predicted_class}")
        else:
            # 2. Model Prediction
            probabilities, predicted_class = model_service.predict(preprocessed_image)
            
            # 3. Generate Grad-CAM Heatmap
            heatmap = xai_service.generate_heatmap(preprocessed_image, predicted_class)
            logger.info(f"[MODEL] Predicted Class: {predicted_class}")
            
        # 4. Generate NLP Explanation
        explanation = nlp_service.explain_heatmap(heatmap)
        
        # 5. Render Visual Overlay
        overlay_pil = overlay_heatmap(original_pil, heatmap, alpha=Config.DEFAULT_ALPHA)
        gradcam_base64 = image_to_base64(overlay_pil)
        
        # Build predictions list for the return object
        prob_list = [float(p) for p in probabilities]
        
        return jsonify({
            "predicted_class": int(predicted_class),
            "probabilities": prob_list,
            "gradcam_image": gradcam_base64,
            "explanation": explanation,
            "is_mock": MOCK_MODE
        }), 200
        
    except Exception as e:
        logger.error(f"Error handling prediction: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred processing your request."}), 500

@app.route("/status", methods=["GET"])
def status():
    """Returns the operational status of the service (indicating if in Mock Mode)."""
    return jsonify({
        "status": "healthy",
        "mock_mode": MOCK_MODE,
        "model_loaded": (model_service is not None and model_service.model is not None)
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
