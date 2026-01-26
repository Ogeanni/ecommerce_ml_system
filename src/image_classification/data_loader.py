"""Image data loading and preprocessing for classification """

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pathlib import Path
import json
import cv2
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

from config import ROOT_DIR, DATA_DIR



class ImageDataLoader:
    """ Load and preprocess image data for classification """
    def __init__(self, dataset_name="fashion_mnist", img_size=(224,224)):
        """ Initialize data loader
        
        Args:
            dataset_name: Name of dataset to load
            img_size: Target image size (height, width)
        """

        self.dataset_name = dataset_name
        self.img_size = img_size
        self.num_classes = None
        self.class_names = None

    def load_fashion_mnist(self) -> Dict:
        """
        Load Fashion MNIST dataset
        
        Returns:
            Dictionary with train/val/test splits
        """

        print("Loading Fashion MNIST dataset...")

        # Load data
        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

        # Class names
        self.class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
        
        self.num_classes = len(self.class_names)

        # Split train into train and validation
        val_size = 10000
        X_train = X_train_full[:-val_size]
        y_train = y_train_full[:-val_size]
        X_val = X_train_full[-val_size:]
        y_val = y_train_full[-val_size:]

        # Normalize to [0, 1]
        X_train = X_train.astype("float32") / 255.0
        X_val = X_val.astype("float32") / 255.0
        X_test = X_test.astype("float32") / 255.0

        # Add channel dimension (grayscale)
        X_train = np.expand_dims(X_train, axis=-1)
        X_val = np.expand_dims(X_val, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        # Resize to target size if needed
        if self.img_size != (28, 28):
            X_train = self._resize_images(X_train, self.img_size)
            X_val = self._resize_images(X_val, self.img_size)
            X_test = self._resize_images(X_test, self.img_size)

        # Convert to RGB (3 channels) for transfer learning
        X_train = np.repeat(X_train, 3, axis=-1)
        X_val = np.repeat(X_val, 3, axis=-1)
        X_test = np.repeat(X_test, 3, axis=-1)

        print(f"   Dataset loaded:")
        print(f"   Training: {X_train.shape}")
        print(f"   Validation: {X_val.shape}")
        print(f"   Test: {X_test.shape}")
        print(f"   Classes: {self.num_classes}")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "class_names": self.class_names,
            "num_classes": self.num_classes
        }
    
    def _resize_images(self, images, target_size):
        """ Resize images to target size """
        resized = np.array([
            cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
            for img in images
        ])

        return resized
    
    def visualize_samples(self, data, num_samples=25, save_path=None):
        """
        Visualize sample images from dataset
        
        Args:
            data: Dictionary with X and y data
            num_samples: Number of samples to show
            save_path: Path to save plot
        """

        X = data["X_train"]
        y = data["y_train"]
        class_names = data["class_names"]

        # Select random samples
        indices = np.random.choice(len(X), num_samples, replace=False)

        # Create grid
        grid_size = int(np.sqrt(num_samples))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12,12))
        axes = axes.flatten()

        for i, idx in enumerate(indices):
            img = X[idx]
            label = y[idx]

            # Display image
            if img.shape[-1] == 1:
                axes[i].imshow(img.squeeze(), cmap="gray")
            else:
                axes[i].imshow(img)

            axes[i].set_title(class_names[label], fontsize=9)
            axes[i].axis("off")
        
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150)

            print(f" Sample visualization saved to {save_path}")

        plt.show()
        plt.close()

    
    def analyze_class_distribution(self, data):
        """ Analyze class distribution in dataset """
        print("\n" + "="*80)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("="*80)

        for split_name in ["train", "val", "test"]:
            y_key = f"y_{split_name}"
            if y_key in data:
                y = data[y_key]
                print(f"\n{split_name.upper()} SET:")

                unique, counts = np.unique(y, return_counts=True)
                for class_idx, count in(unique, counts):
                    class_name = data["class_names"][class_idx]
                    percentage = count / len(y) * 100

                    print(f"  {class_name:20s}: {count:5d} ({percentage:5.2f}%)")

def load_image_data(dataset_name="fashion_mnist", img_size=(224, 224)):
    """
    Convenience function to load image data
    
    Args:
        dataset_name: Name of dataset
        img_size: Target image size
        
    Returns:
        Dictionary with data splits
    """

    loader =  ImageDataLoader(dataset_name=dataset_name, img_size=img_size)
    if dataset_name == "fashion_mnist":
        data = loader.load_fashion_mnist()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Visualize samples# Visualize samples
    loader.visualize_samples(data, save_path=f'results/image_classification/sample_images_{dataset_name}.png')

    # Analyze distribution
    loader.analyze_class_distribution(data)

    return data

   
   
if __name__ == "__main__":
    # Test data loading
    data = load_image_data("fashion_mnist", img_size=(224, 224))