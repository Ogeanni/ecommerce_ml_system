"""
Convert existing H5 model to SavedModel format
"""
import tensorflow as tf
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Path to your existing H5 model
h5_model_path = "models/image_classification/saved_models/fashion_mnist_transfer_final.h5"

print(f"TensorFlow version: {tf.__version__}")
print(f"\nLoading H5 model from: {h5_model_path}")

try:
    # Load the H5 model
    model = tf.keras.models.load_model(h5_model_path)
    print(" Model loaded successfully")
    
    # Print model summary
    print("\nModel Summary:")
    model.summary()
    
    # Create SavedModel directory path
    h5_path = Path(h5_model_path)
    savedmodel_dir = h5_path.parent / h5_path.stem  # Remove .h5 extension
    
    print(f"\nSaving model in SavedModel format to: {savedmodel_dir}")
    
    # Save in SavedModel format
    model.save(str(savedmodel_dir), save_format='tf')
    print(f" Model saved successfully!")
    
    # Copy class names if they exist
    class_names_src = h5_path.parent / "class_names.json"
    if class_names_src.exists():
        import shutil
        class_names_dst = savedmodel_dir / "class_names.json"
        shutil.copy(str(class_names_src), str(class_names_dst))
        print(f" Class names copied to {class_names_dst}")
    else:
        # Create default class names for Fashion MNIST
        class_names = [
            'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
        ]
        class_names_path = savedmodel_dir / "class_names.json"
        with open(class_names_path, 'w') as f:
            json.dump(class_names, f, indent=2)
        print(f" Default class names created at {class_names_path}")
    
    # Copy training history if it exists
    history_src = h5_path.parent / f"{h5_path.stem}_history.json"
    if history_src.exists():
        import shutil
        history_dst = savedmodel_dir / "training_history.json"
        shutil.copy(str(history_src), str(history_dst))
        print(f" Training history copied to {history_dst}")
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"\nYour new model path:")
    print(f'model_path = "{savedmodel_dir}"')
    print(f"\nUpdate your predict.py to use this path instead of the .h5 file")
    
except Exception as e:
    print(f"\n Error: {e}")
    print("\nMake sure:")
    print("1. The model file exists at the specified path")
    print("2. You're using the same TensorFlow version that created the model")
    print("3. All dependencies are installed")