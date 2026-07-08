import tensorflow as tf
from keras.models import Model
import numpy as np

class XaiService:
    def __init__(self, model, last_conv_layer_name):
        self.model = model
        self.last_conv_layer_name = last_conv_layer_name
        self._validate_layer()
        
    def _validate_layer(self):
        """Verify the specified convolutional layer exists in the model."""
        try:
            self.model.get_layer(self.last_conv_layer_name)
        except ValueError as e:
            layer_names = [l.name for l in self.model.layers[-10:]]
            raise ValueError(
                f"Layer '{self.last_conv_layer_name}' not found in the model. "
                f"Some available trailing layers: {layer_names}"
            ) from e

    def generate_heatmap(self, img_array, predicted_class):
        """
        Generates a 2D Grad-CAM heatmap array for the given image and target class.
        """
        # Create a sub-model that outputs both the target conv feature map and final predictions
        grad_model = Model(
            inputs=self.model.inputs,
            outputs=[self.model.get_layer(self.last_conv_layer_name).output, self.model.output]
        )
        
        # Track gradients using GradientTape
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            
            # Handle potential list/tuple wrapping of outputs from Model call
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[0]
            if isinstance(conv_outputs, (list, tuple)):
                conv_outputs = conv_outputs[0]
                
            class_channel = predictions[:, predicted_class]
            
        # Get gradients of the predicted class score with respect to feature maps
        grads = tape.gradient(class_channel, conv_outputs)
        
        # Compute the channel-wise mean weight
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the feature channels by their gradient weights
        conv_outputs_batch = conv_outputs[0]
        heatmap = conv_outputs_batch @ pooled_grads[..., tf.newaxis]
        
        # Squeeze dimensions and apply ReLU
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0)
        
        # Normalize the heatmap to the [0, 1] range
        max_val = tf.math.reduce_max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val
            
        return heatmap.numpy()
