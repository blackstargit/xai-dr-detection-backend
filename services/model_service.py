import logging
import tensorflow as tf
from keras.models import load_model
import numpy as np

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self._configure_gpu_memory()
        self.load_model()
        
    def _configure_gpu_memory(self):
        """Configure TensorFlow to grow GPU memory dynamically if GPU is present."""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logger.info("Configured dynamic GPU memory growth.")
            except Exception as e:
                logger.warning(f"Error configuring GPU memory growth: {e}")
                
    def load_model(self):
        """Loads the ResNet50 model from the specified path."""
        try:
            logger.info(f"Loading ResNet50 model from {self.model_path}...")
            self.model = load_model(self.model_path)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load Keras model from {self.model_path}: {e}")
            raise e
            
    def predict(self, preprocessed_image):
        """
        Runs model inference on preprocessed image.
        Returns:
            probabilities: A numpy array of class probabilities.
            predicted_class: The integer predicted class.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot perform prediction.")
            
        predictions = self.model.predict(preprocessed_image)
        probabilities = predictions[0]
        predicted_class = np.argmax(probabilities)
        return probabilities, predicted_class
