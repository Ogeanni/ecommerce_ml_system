"""
Inference pipeline for image classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.utils import custom_object_scope
import cv2
from pathlib import Path
import json
from typing import Union, List, Tuple
import matplotlib.pyplot as plt


class ImageClassificationPredictor:
    """Make predictions on new images"""
    def __init__(self, model_path, img_size=(224, 224)):
        """
        Initialize predictor
        
        Args:
            model_path: Path to saved model (.h5 file)
            img_size: Expected input image size
        """
        print(f"Loading model from {model_path}...")

        model_path_obj = Path(model_path)

        if model_path_obj.is_dir():
            # SavedModel format
            self.model = tf.keras.models.load_model(str(model_path))
            print(" Loaded SavedModel format")
        else:
            # H5 format - load with compile=False to avoid TFOpLambda issues
            try:
                self.model = tf.keras.models.load_model(str(model_path), compile=False)
                print(" Loaded H5 format")
            except Exception as e:
                print(f" Error loading model: {e}")
                raise

        self.img_size = img_size

        if model_path_obj.is_dir():
            class_names_path = model_path_obj / "class_names.json"
        else:
            class_names_path = model_path_obj.parent / "class_names.json"

        if class_names_path.exists():
            with open(class_names_path, "r") as f:
                self.class_names = json.load(f)
            print(f" Loaded class names from {class_names_path}")
        else:
            # Default class names (Fashion MNIST)
            self.class_names = [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
            ]
            print(" Using default Fashion MNIST class names")
            
        print(f" Model loaded successfully")
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Expected input size: {self.img_size}")


    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess image for prediction
        
        Args:
            image: Path to image file or numpy array
            
        Returns:
            Preprocessed image
        """
        # Load image if path provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()

        # Resize
        img = cv2.resize(img, self.img_size)

        # Convert to float and normalize
        img = img.astype("float32")/255.0

        # Ensure 3 channels
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        return img
    
    def predict_single(self, image: Union[str, np.ndarray], top_k = 3) -> dict:
        """
        Predict class for a single image
        
        Args:
            image: Path to image or numpy array
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        # Preprocess
        img = self.preprocess_image(image)

        # Add batch dimension
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        predictions = self.model.predict(img_batch, verbose=0)[0]

        # Get top-k predictions
        top_indices = np.argsort(predictions)[::-1][:top_k]

        results = {
            "top_predictions": [],
            "all_probabilities": predictions.tolist()
        }
        for idx in top_indices:
            results["top_predictions"].append({
                "class": self.class_names[idx],
                "class_id": int(idx),
                "probability": float(predictions[idx]),
                "confidence": f"{predictions[idx]*100:.2f}%"
            })
        
        # Best prediction
        best_idx = top_indices[0]
        results["predicted_class"] = self.class_names[best_idx]
        results["predicted_class_id"] = int(best_idx)
        results["confidence"] = float(predictions[best_idx])

        return results
    
    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[dict]:
        """
        Predict classes for multiple images
        
        Args:
            images: List of image paths or numpy arrays
            
        Returns:
            List of prediction dictionaries
        """
        print(f"Predicting {len(images)} images...")
        
        # Preprocess all images
        img_batch = np.array([self.preprocess_image(img) for img in images])

        # Predict
        predictions = self.model.predict(img_batch, verbose=0)

        # Format results
        results = []
        for i, preds in enumerate(predictions):
            best_idx = np.argmax(preds)
            results.append({
                "image_index": i,
                "predicted_class": self.class_names[best_idx],
                "predicted_class_id": int(best_idx),
                "confidence": float(preds[best_idx])
            })

        return results
    
def predict_image(model_path: str, image_path: str, visualize=False):
    """
    Convenience function to predict single image
    
    Args:
        model_path: Path to saved model
        image_path: Path to image
        visualize: Whether to show visualization
        
    Returns:
        Prediction results
    """
    predictor = ImageClassificationPredictor(model_path)
    result = predictor.predict_single(image_path)

    return result

# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    print("="*80)
    print("IMAGE CLASSIFICATION - PREDICTION EXAMPLE")
    print("="*80)
    
    # Example: Load model and predict
    savedmodel_path = "models/image_classification/saved_models/fashion_mnist_transfer_final"
    h5_path = "models/image_classification/saved_models/fashion_mnist_transfer_final.h5"

     # Determine which model exists
    if Path(savedmodel_path).exists():
        model_path = savedmodel_path
        print(f"\n Found SavedModel: {model_path}")
    elif Path(h5_path).exists():
        model_path = h5_path
        print(f"\n Found H5 model: {model_path}")
    else:
        print(f"\n No model found at:")
        print(f"   - {savedmodel_path}")
        print(f"   - {h5_path}")
        print("\n Train a model first using scripts/train_image_classifier.py")
        exit(1)

    # Load predictor
    predictor = ImageClassificationPredictor(model_path)

    # Load test data to get sample images
    try:
        from src.image_classification.data_loader import load_image_data
        data = load_image_data('fashion_mnist', img_size=(224, 224))

        # Predict on a few test images
        for i in range(5):
            test_image = data["X_test"][i]
            true_label = data["class_names"][data["y_test"][i]]

            print(f"\n{'='*80}")
            print(f"Sample {i+1}")
            print(f"True label: {true_label}")
            print(f"{'='*80}")

            result = predictor.predict_single(test_image)

            print(f"\nTop 3 Predictions:")
            for pred in result['top_predictions']:
                correct = "Good" if pred['class'] == true_label else "Bad"
                print(f"  {correct} {pred['class']:<20} {pred['confidence']}")

    except ImportError:
        print("\n  Could not import data_loader. Testing with random image instead...")
        
        # Create random test image
        random_image = np.random.rand(28, 28, 3).astype('float32')
        result = predictor.predict_single(random_image)
        
        print(f"\nPrediction on random image:")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")