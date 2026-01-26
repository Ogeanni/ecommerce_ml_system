"""
Memory-safe image data loading for classification
(Fashion-MNIST compatible with transfer learning)
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np



class DatasetWrapper:
    """
    Thin wrapper so existing code using data["X_train"] does NOT break.
    Wraps tf.data.Dataset safely.
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        return iter(self.dataset)

    def take(self, n):
        return DatasetWrapper(self.dataset.take(n))

    def as_tf_dataset(self):
        return self.dataset


class ImageDataLoader:
    """Load and preprocess image datasets safely (no OOM)"""

    def __init__(self, dataset_name="fashion_mnist", img_size=(224, 224), batch_size=32):
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = None
        self.num_classes = None

    def _preprocess(self, x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.expand_dims(x, -1)                     # (H, W, 1)
        x = tf.image.resize(x, self.img_size)         # resize lazily
        x = tf.image.grayscale_to_rgb(x)               # (H, W, 3)
        return x, y

    def _make_dataset(self, x, y, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((x, y))
        if shuffle:
            ds = ds.shuffle(10_000)
        ds = (
            ds.map(self._preprocess, num_parallel_calls=tf.data.AUTOTUNE)
              .batch(self.batch_size)
              .prefetch(tf.data.AUTOTUNE)
        )
        return ds

    def load_fashion_mnist(self):
        print("Loading Fashion MNIST dataset (memory-safe)...")

        (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

        self.class_names = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        ]
        self.num_classes = len(self.class_names)

        # Train / validation split
        val_size = 10_000
        X_train, y_train = X_train_full[:-val_size], y_train_full[:-val_size]
        X_val, y_val     = X_train_full[-val_size:], y_train_full[-val_size:]

        # Create datasets (no augmentation yet)
        train_dataset = self._make_dataset(X_train, y_train, shuffle=True)
        val_dataset   = self._make_dataset(X_val, y_val, shuffle=False)
        test_dataset  = self._make_dataset(X_test, y_test, shuffle=False)

        return {
            "X_train": train_dataset,
            "y_train": train_dataset,
            "X_val": val_dataset,
            "y_val": val_dataset,
            "X_test": test_dataset,
            "y_test": test_dataset,
            "X_train_raw": X_train,
            "y_train_raw": y_train,
            "X_val_raw": X_val,
            "y_val_raw": y_val,
            "X_test_raw": X_test,
            "y_test_raw": y_test,
            "class_names": self.class_names,
            "num_classes": self.num_classes
        }
        

def load_image_data_2(dataset_name="fashion_mnist", img_size=(224, 224), batch_size=32):
    """
    Convenience function (keeps your existing API unchanged)
    """
    loader = ImageDataLoader(
        dataset_name=dataset_name,
        img_size=img_size,
        batch_size=batch_size
    )

    if dataset_name == "fashion_mnist":
        return loader.load_fashion_mnist()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
if __name__ == "__main__":
    # Test data loading
    data = load_image_data_2("fashion_mnist", img_size=(224, 224))
