import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
import cv2

from scripts.nlp import DiabeticRetinopathyExplainer


# --- 5. Preprocessing Function for a Single Image ---
def preprocess_image_for_inference(image_path, image_size):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # Use ResNet50 preprocess_input
    return img_array


# --- 6. Grad-CAM Implementation ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def display_gradcam(img_path, heatmap, predicted_class, alpha=0.4):
    # Explain the heatmap
    explainer = DiabeticRetinopathyExplainer(
        threshold=0.7
    )  # Adjust threshold if needed
    activated_regions = explainer.analyze_heatmap_regions(heatmap)
    explanation = explainer.generate_explanation(activated_regions)

    print(f"Explaination: {explanation}")
    img = cv2.imread(img_path)

    heatmap = np.uint8(255 * heatmap)
    jet = plt.colormaps.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Display Grad CAM
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Original Image
    axes[0].imshow(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    )  # Convert BGR to RGB for display
    # axes[0].set_title("Original Image")
    # Title spanning both subplots
    fig.suptitle(
        "Grad-CAM Heatmap\n"
        + f"Predicted Class: {predicted_class}\n"
        + f"Explanation: {explanation}",
        fontsize=14,  # Adjust font size as needed
    )

    axes[0].axis("off")

    # Grad-CAM Heatmap
    axes[1].imshow(superimposed_img)

    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
