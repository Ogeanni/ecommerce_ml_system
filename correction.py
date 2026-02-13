"""
Complete end-to-end training script for image classification
"""

import sys
from pathlib import Path

# For running scripts on my local computer
#PROJECT_ROOT = Path(__file__).resolve().parent.parent
#sys.path.append(str(PROJECT_ROOT))


from config import ROOT_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR


import tensorflow as tf
import numpy as np
from pathlib import Path

# Import custom modules
from src.image_classification.data_loader import load_image_data
from src.image_classification.data_loader_2 import load_image_data_2
from src.image_classification.augmentation import create_augmentation_layer, create_tf_dataset
from src.image_classification.model_builder import build_model
from src.image_classification.train import ImageClassificationTrainer
from src.image_classification.evaluate import ImageClassificationEvaluator

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("="*80)
print("IMAGE CLASSIFICATION - COMPLETE TRAINING PIPELINE")
print("="*80)


# ==================== CONFIGURATION ====================
CONFIG = {
    # Data
    'dataset_name': 'fashion_mnist',  
    'img_size': (124, 124),                     # Image size
    
    # Model
    'model_type': 'transfer',                     # 'simple', 'transfer', 'residual'
    'base_model_name': "mobilenetv2",                    # For transfer learning
    
    # Training
    'batch_size': 32,
    'epochs': 2,
    'learning_rate': 0.001,
    'optimizer': 'adam',
    
    # Augmentation
    'use_augmentation': False,
    
    # Callbacks
    'early_stopping_patience': 10,
    'reduce_lr_patience': 5,
    
    # Fine-tuning (for transfer learning)
    'fine_tune': False,
    'fine_tune_epochs': 10,
    'fine_tune_lr': 1e-5,
}

print("\nConfiguration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

# ==================== STEP 1: LOAD DATA ====================
print("\n" + "="*80)
print("STEP 1: LOADING DATA")
print("="*80)

data = load_image_data_2(
    dataset_name=CONFIG['dataset_name'],
    img_size=CONFIG['img_size']
)

print(f"\nData loaded successfully!")
print(f"  Classes: {data['class_names']}")
print(f"  Number of classes: {data['num_classes']}")

train_dataset = data["X_train"]
val_dataset   = data["X_val"]
test_dataset  = data["X_test"]

# ==================== STEP 2: CREATE DATA AUGMENTATION ====================
# if CONFIG['use_augmentation']:
#     print("\n" + "="*80)
#     print("STEP 2: CREATING DATA AUGMENTATION")
#     print("="*80)
    
#     augmentation_layer = create_augmentation_layer()
    
# # Create TensorFlow datasets

#     train_dataset = create_tf_dataset(data["X_train"], 
#                                       data["y_train"], 
#                                       batch_size=CONFIG["batch_size"],
#                                       augment=True,
#                                       shuffle=True)

#     val_dataset = create_tf_dataset(data["X_val"], 
#                                     data["y_val"], 
#                                     batch_size=CONFIG["batch_size"],
#                                     augment=False,
#                                     shuffle=False)
    
#     print(" Data augmentation pipeline created")
# else:
#     train_dataset = None
#     val_dataset = None
#     print("\nSkipping data augmentation")



# ==================== Try this AUGMENTATION ====================
if CONFIG['use_augmentation']:
    augmentation_layer = create_augmentation_layer()

    # Apply augmentation only on training dataset
    train_dataset = train_dataset.map(
        lambda x, y: (augmentation_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

# Prefetch for performance
train_dataset = train_dataset.prefetch(1)
val_dataset   = val_dataset.prefetch(1)
test_dataset  = test_dataset.prefetch(1)
print(type(train_dataset))
print(train_dataset.element_spec)



# ==================== STEP 3: BUILD MODEL ====================

print("\n" + "="*80)
print("STEP 3: BUILDING MODEL")
print("="*80)

if CONFIG["model_type"] == "transfer":
    model = build_model(model_type = "transfer",
                        input_shape =(*CONFIG["img_size"], 3),
                        num_classes = data["num_classes"],
                        base_model_name = CONFIG["base_model_name"],
                        trainable_layers=0  # Freeze base model initially
                        )
else:
    model = build_model(model_type=CONFIG["model_type"],
                        input_shape=(*CONFIG["img_size"], 3),
                        num_classes=data["num_classes"])
    
print("\nModel Architecture:")
model.summary()

# Count parameters
total_params = model.count_params()
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable_params = total_params - trainable_params

print(f"\nModel Parameters:")
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Non-trainable parameters: {non_trainable_params:,}")

# ==================== STEP 4: INITIALIZE TRAINER ====================
print("\n" + "="*80)
print("STEP 4: INITIALIZING TRAINER")
print("="*80)

trainer = ImageClassificationTrainer(
    model=model,
    data=data,
    model_name=f"{CONFIG['dataset_name']}_{CONFIG['model_type']}"
)

# Compile model
trainer.compile_model(
    optimizer=CONFIG['optimizer'],
    learning_rate=CONFIG['learning_rate']
)

# Create callbacks
callbacks_list = trainer.create_callbacks(
    early_stopping_patience=CONFIG['early_stopping_patience'],
    reduce_lr_patience=CONFIG['reduce_lr_patience']
)

# ==================== STEP 5: TRAIN MODEL ====================
print("\n" + "="*80)
print("STEP 5: TRAINING MODEL")
print("="*80)

if train_dataset is not None:
    # Train with augmented datasets
    history = model.fit(
        train_dataset,
        epochs=CONFIG['epochs'],
        validation_data=val_dataset,
        callbacks=callbacks_list,
        verbose=1
    )
    trainer.history = type('History', (), {'history': history.history})()
else:
    # Train without augmentation
    history = trainer.train(
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size'],
        callbacks_list=callbacks_list
    )


# ==================== STEP 6: FINE-TUNING (Optional) ====================
if CONFIG['model_type'] == 'transfer' and CONFIG['fine_tune']:
    print("\n" + "="*80)
    print("STEP 6: FINE-TUNING MODEL")
    print("="*80)
    
    trainer.fine_tune(
        base_model_layers_to_unfreeze=20,
        epochs=CONFIG['fine_tune_epochs'],
        learning_rate=CONFIG['fine_tune_lr']
    )
    
    # Plot updated training history
    trainer.plot_training_history(
        save_path=RESULTS_DIR /'training_history_with_finetune.png'
    )

# ==================== STEP 7: SAVE MODEL ====================
print("\n" + "="*80)
print("STEP 7: SAVING MODEL")
print("="*80)

trainer.save_model(save_format='tf') 

# ==================== STEP 8: EVALUATE MODEL ====================
print("\n" + "="*80)
print("STEP 8: EVALUATING MODEL")
print("="*80)

evaluator = ImageClassificationEvaluator(
    model=model,
    data=data,
    class_names=data['class_names']
)

# Generate comprehensive evaluation report
evaluator.generate_full_report(
    dataset='test',
    output_dir=RESULTS_DIR
)


# ==================== STEP 9: FINAL SUMMARY ====================
print("\n" + "="*80)
print("TRAINING PIPELINE COMPLETE!")
print("="*80)

# Evaluate on all datasets
print("\nFinal Performance Summary:")
print("-" * 80)

for dataset in ['train', 'val', 'test']:
    metrics = evaluator.evaluate(dataset=dataset, verbose=0)
    print(f"\n{dataset.upper()} SET:")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value:.4f}")

print("\n" + "="*80)
print("FILES SAVED:")
print("="*80)
print(f"  Model: models/image_classification/saved_models/")
print(f"  Checkpoints: models/image_classification/checkpoints/")
print(f"  Logs: models/image_classification/logs/")
print(f"  Results: results/image_classification/")
print("\n All done!")








def save_model(self, save_path=None, save_format='tf'):
        """ Save trained model
        
        Args:
            save_path: Path to save model 
        """
        if save_path is None:
            if save_format == "tf":
                # SavedModel format uses a directory
                save_path = self.model_dir / f'{self.model_name}_final'
            else:
                # H5 format uses a file
                save_path = self.model_dir / f'{self.model_name}_final.h5'

        print(f"\nSaving model in {save_format} format to {save_path}...")

        if save_format == 'tf':
            # Save in SavedModel format (directory)
            self.model.save(str(save_path), save_format='tf')
            print(f" Model saved to {save_path}/")

            # Also save class names if available
            if 'class_names' in self.data:
                class_names_path = Path(save_path) / 'class_names.json'
                with open(class_names_path, 'w') as f:
                    json.dump(self.data['class_names'], f, indent=2)
                print(f" Class names saved to {class_names_path}")

        else:
            # Save in H5 format (single file)
            self.model.save(str(save_path), save_format='h5')
            print(f" Model saved to {save_path}")
        
        # Save training history
        if self.history:
            if save_format == 'tf':
                history_path = Path(save_path) / 'training_history.json'
            else:
                history_path = Path(save_path).parent / f'{self.model_name}_history.json'
            
            with open(history_path, 'w') as f:
                # Convert numpy values to python types
                history_dict = {
                key: [float(val) for val in values] 
                for key, values in self.history.history.items()
            }
                json.dump(history_dict, f, indent=2)
            print(f" Training history saved to {history_path}")













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

        # Predict - handle both regular models and exported models
        if self._is_exported_model:
            # For exported SavedModel, need to convert to tensor
            import tensorflow as tf
            img_tensor = tf.convert_to_tensor(img_batch, dtype=tf.float32)
            
            # Get the input key (usually the first key)
            input_key = list(self.model.structured_input_signature[1].keys())[0]
            
            # Run inference
            output = self.model(**{input_key: img_tensor})
            
            # Get the output key (usually the first key)
            output_key = list(output.keys())[0]
            predictions = output[output_key].numpy()[0]
        else:
            # Regular Keras model
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
