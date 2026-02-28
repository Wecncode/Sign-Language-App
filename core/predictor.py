"""
=============================================================================
PREDICTOR MODULE - Simple Keras Version (Latest TensorFlow)
=============================================================================
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np
import pickle
from config import Config


class Predictor:
    """
    Loads and runs the trained Keras model.
    Works with TensorFlow 2.16+ and latest MediaPipe.
    """
    
    def __init__(self, model_path=None, labels_path=None):
        """
        Initialize the Predictor.
        """
        
        # Set default paths
        if model_path is None:
            model_path = os.path.join(Config.MODEL_DIR, "best_model.keras")
        
        if labels_path is None:
            labels_path = Config.LABELS_PATH
        
        # Check files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found: {model_path}\n"
                f"Download 'best_model.keras' from Google Drive."
            )
        
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                f"Labels not found: {labels_path}\n"
                f"Download 'labels.pkl' from Google Drive."
            )
        
        # Load labels
        print(f"Loading labels from: {labels_path}")
        with open(labels_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.num_classes = len(self.label_encoder)
        self.class_names = [self.label_encoder[i] for i in range(self.num_classes)]
        
        # Load model
        print(f"Loading model from: {model_path}")
        self.model = tf.keras.models.load_model(model_path)
        
        # Get input shape
        input_shape = self.model.input_shape
        self.sequence_length = input_shape[1]
        self.features_per_frame = input_shape[2]
        
        # Find neutral index
        self.neutral_index = self._find_neutral_index()
        
        # Print summary
        print(f"\n✓ Predictor initialized!")
        print(f"  Classes: {self.class_names}")
    
    def _find_neutral_index(self):
        for index, label in self.label_encoder.items():
            if label == Config.NEUTRAL_LABEL:
                return index
        return None
    
    def predict(self, sequence):
        """Run prediction on a sequence."""
        sequence = np.array(sequence)
        input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
        
        output = self.model.predict(input_data, verbose=0)
        probabilities = output[0]
        
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index]
        predicted_class = self.label_encoder[predicted_index]
        
        return predicted_class, confidence, probabilities
    
    def predict_top_k(self, sequence, k=3):
        _, _, probabilities = self.predict(sequence)
        top_k_indices = np.argsort(probabilities)[::-1][:k]
        return [(self.label_encoder[i], probabilities[i]) for i in top_k_indices]
    
    def is_neutral(self, predicted_class):
        return predicted_class == Config.NEUTRAL_LABEL
    
    def get_class_names(self):
        return self.class_names.copy()
    
    def get_num_classes(self):
        return self.num_classes
    
    def get_model_info(self):
        return {
            'sequence_length': self.sequence_length,
            'features_per_frame': self.features_per_frame,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
        }


# =============================================================================
# TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("PREDICTOR TEST")
    print("=" * 60)
    
    try:
        predictor = Predictor()
        print("✓ Predictor loaded!")
        
        # Test prediction
        fake_data = np.random.randn(30, 126).astype(np.float32)
        sign, conf, _ = predictor.predict(fake_data)
        print(f"\nTest prediction: {sign} ({conf:.1%})")
        
        # Speed test
        import time
        start = time.time()
        for _ in range(10):
            predictor.predict(np.random.randn(30, 126).astype(np.float32))
        elapsed = time.time() - start
        print(f"Speed: {10/elapsed:.1f} predictions/sec")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Error: {e}")