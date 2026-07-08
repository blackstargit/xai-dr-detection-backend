import sys
import os
from PIL import Image, ImageDraw
import io
import numpy as np

# Add parent directory to sys.path so we can import app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from services.model_service import ModelService
from services.xai_service import XaiService
from services.nlp_service import NlpService
from utils.image_utils import preprocess_image_from_bytes, overlay_heatmap, image_to_base64

def test_pipeline():
    print("Starting pipeline verification test...")
    
    # 1. Create a dummy retinal image (a 300x300 red/brown canvas with a mock blood vessel circle)
    print("Creating dummy retinal image...")
    img = Image.new("RGB", (300, 300), color=(180, 50, 20))
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 80, 220, 220], outline=(100, 0, 0), width=4)
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_bytes = img_byte_arr.getvalue()
    
    # 2. Test Image Preprocessing
    print("Testing image preprocessing...")
    preprocessed_image, original_pil = preprocess_image_from_bytes(img_bytes, Config.IMAGE_SIZE)
    assert preprocessed_image.shape == (1, 224, 224, 3), f"Invalid preprocessed shape: {preprocessed_image.shape}"
    assert original_pil.size == (300, 300), f"Invalid original PIL size: {original_pil.size}"
    print("Image preprocessing passed!")
    
    # 3. Test Model Service Loading
    print(f"Testing model service loading from path: {Config.MODEL_PATH}...")
    try:
        model_service = ModelService(Config.MODEL_PATH)
        print("Model loaded successfully!")
        
        # 4. Test Prediction
        print("Testing model prediction...")
        probabilities, predicted_class = model_service.predict(preprocessed_image)
        print(f"Probabilities: {probabilities}")
        print(f"Predicted Class: {predicted_class} ({Config.CLASS_LABELS[predicted_class]})")
        assert len(probabilities) == 5, f"Expected 5 classes, got {len(probabilities)}"
        assert 0 <= predicted_class < 5, f"Invalid predicted class: {predicted_class}"
        print("Model prediction passed!")
        
        # 5. Test XAI Grad-CAM Heatmap
        print("Testing XAI Grad-CAM generation...")
        xai_service = XaiService(model_service.model, Config.LAST_CONV_LAYER_NAME)
        heatmap = xai_service.generate_heatmap(preprocessed_image, predicted_class)
        assert len(heatmap.shape) == 2, f"Invalid heatmap shape: {heatmap.shape}"
        print(f"Heatmap shape: {heatmap.shape}")
        print("XAI Grad-CAM passed!")
        
        # 6. Test NLP Explainer
        print("Testing NLP Explainer...")
        nlp_service = NlpService(Config.EXPLANATION_THRESHOLD)
        explanation = nlp_service.explain_heatmap(heatmap)
        print(f"Explanation: {explanation}")
        assert isinstance(explanation, str) and len(explanation) > 0, "Invalid explanation text"
        print("NLP Explainer passed!")
        
        # 7. Test Heatmap Overlay & Base64 packing
        print("Testing heatmap overlay blending...")
        overlay_pil = overlay_heatmap(original_pil, heatmap, alpha=Config.DEFAULT_ALPHA)
        assert overlay_pil.size == original_pil.size, "Overlay size mismatch"
        base64_str = image_to_base64(overlay_pil)
        assert len(base64_str) > 0, "Base64 encoding empty"
        print("Overlay and encoding passed!")
        
    except Exception as e:
        import traceback
        print("\n[FAIL] Test encountered error during deep learning execution:")
        traceback.print_exc()
        print("Please verify your TensorFlow/Keras environment or check if model weights are corrupt.")
        sys.exit(1)
        
    print("\n[SUCCESS] All pipeline checks passed successfully!")

if __name__ == "__main__":
    test_pipeline()
