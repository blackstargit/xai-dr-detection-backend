import io
import base64
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.applications.resnet50 import preprocess_input

def preprocess_image_from_bytes(image_bytes, target_size):
    """
    Reads image bytes, standardizes it to RGB mode, resizes to target_size,
    and returns both the preprocessed tensor array for models and the original PIL image.
    """
    original_pil = Image.open(io.BytesIO(image_bytes))
    if original_pil.mode != "RGB":
        original_pil = original_pil.convert("RGB")
        
    resized_pil = original_pil.resize(target_size)
    
    # Convert to array and expand dims
    img_array = np.array(resized_pil, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply ResNet50 preprocessing
    preprocessed_tensor = preprocess_input(img_array)
    
    return preprocessed_tensor, original_pil

def overlay_heatmap(original_pil, heatmap, alpha=0.4):
    """
    Applies the Jet colormap to a heatmap and overlays it on the original PIL image.
    Returns a blended PIL Image.
    """
    # Safety: ensure heatmap is in [0, 1] range
    if heatmap.max() > 0:
        normalized_heatmap = heatmap / heatmap.max()
    else:
        normalized_heatmap = heatmap

    # Scale to 0-255
    heatmap_255 = np.uint8(255 * normalized_heatmap)

    # Apply Matplotlib jet colormap
    colormap = plt.colormaps.get_cmap("jet")
    colorized_heatmap = colormap(np.arange(256))[:, :3]  # Extract RGB channels
    jet_heatmap = colorized_heatmap[heatmap_255]          # Map values

    # Convert to PIL Image and resize to match original image dimensions
    jet_heatmap_img = Image.fromarray(np.uint8(255 * jet_heatmap))
    jet_heatmap_img = jet_heatmap_img.resize(original_pil.size)
    
    # Convert both back to numpy arrays for pixel math
    original_array = np.array(original_pil, dtype=np.float32)
    jet_array = np.array(jet_heatmap_img, dtype=np.float32)
    
    # Blending formula
    blended_array = jet_array * alpha + original_array
    blended_array = np.clip(blended_array, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blended_array)

def image_to_base64(pil_image):
    """
    Converts a PIL Image into a base64 encoded string.
    """
    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    return base64.b64encode(image_bytes).decode("utf-8")
