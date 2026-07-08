import os

# Compute the base directory of the backend package (where config.py resides)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class Config:
    # Model Configurations
    MODEL_PATH = os.environ.get(
        "MODEL_PATH", 
        os.path.join(BASE_DIR, "assets", "best_resnet50_model.keras")
    )
    LAST_CONV_LAYER_NAME = os.environ.get("LAST_CONV_LAYER_NAME", "conv5_block3_out")
    IMAGE_SIZE = (224, 224)
    NUM_CLASSES = 5

    # XAI Configurations
    DEFAULT_ALPHA = 0.4
    EXPLANATION_THRESHOLD = 0.7

    # Diagnostic Class Mappings
    CLASS_LABELS = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }
