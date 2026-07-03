from scripts.nlp import DiabeticRetinopathyExplainer
from scripts.xai import (
    preprocess_image_for_inference,
    make_gradcam_heatmap,
    display_gradcam,
)
from scripts.resnet50 import load_resnet_model
import numpy as np

# Configuration
image_size = (224, 224)
batch_size = 32
num_classes = 5
learning_rate = 1e-4

model_path = "../ml_resnet_xai/best_resnet50_model.keras"
last_conv_layer_name = "conv5_block3_out"
image_path = (
    "../ml_resnet_xai/archive/colored_images/Severe/0104b032c141.png"  # Replace with your image path
)


def main():
    # Process image for prediction
    preprocessed_image = preprocess_image_for_inference(image_path, image_size)

    # Load model
    model = load_resnet_model(model_path)
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])

    print(f"Predicted class: {predicted_class}")
    print(f"Probabilities: {predictions}")

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(
        preprocessed_image, model, last_conv_layer_name, pred_index=predicted_class
    )

    # Display Grad-CAM
    display_gradcam(image_path, heatmap, predicted_class=predicted_class)

    # Explain the heatmap
    explainer = DiabeticRetinopathyExplainer(
        threshold=0.7
    )  # Adjust threshold if needed
    activated_regions = explainer.analyze_heatmap_regions(heatmap)
    explanation = explainer.generate_explanation(activated_regions)

    print(f"Explaination: {explanation}")

if __name__ == "__main__":
    main()
