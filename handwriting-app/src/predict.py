import tensorflow as tf
import numpy as np
import string

class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        # Mapping 0-25 to A-Z
        self.labels = list(string.ascii_uppercase)

    def predict(self, processed_image):
        if processed_image is None:
            return None, 0.0

        predictions = self.model.predict(processed_image)
        predicted_index = np.argmax(predictions)
        confidence = np.max(predictions)
        
        predicted_letter = self.labels[predicted_index]
        
        return predicted_letter, float(confidence)
