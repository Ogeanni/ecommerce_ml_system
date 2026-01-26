""" Data augmentation for image classification """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np



def create_augmentation_layer(include_cutout=False):
    """
    Create data augmentation layer
    
    Args:
        include_cutout: Whether to include random cutout/erasing
        
    Returns:
        Sequential model with augmentation layers
    """

    augmentation_layers = [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),             # Rotate up to 10%
        layers.RandomZoom(0.1),                 # Zoom up to 10%
        layers.RandomTranslation(0.1, 0.1),     # Translate up to 10%
        layers.RandomContrast(0.2)              # Adjust contrast
    ]

    return keras.Sequential(augmentation_layers, name="augmentation")


class DataGenerator:
    """ Generate custom batches with augmentation """

    def __init__(self, X, y, batch_size=32, augment=True, shuffle=True):
        """
        Initialize data generator
        
        Args:
            X: Images
            y: Labels
            batch_size: Batch size
            augment: Whether to apply augmentation
            shuffle: Whether to shuffle data
        """
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.num_samples = len(X)
        self.indices = np.arange(self.num_samples)

        if augment:
            self.augmentation = create_augmentation_layer()

        if shuffle:
            np.random.shuffle(self.indices)


    def __len__(self):
        """ Number of batches per epoch """
        return int(np.ceil(self.num_samples / self.batch_size))
    
    def __getitem__(self, idx):
        """ Get batch """
        # Get batch indices
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.num_samples)
        batch_indices = self.indices[start_idx:end_idx]

        # Get batch data
        X_batch = self.X[batch_indices]
        y_batch = self.y[batch_indices]

        # Apply augmentation
        if self.augment:
            X_batch = self.augmentation(X_batch, training=True)

        return X_batch, y_batch
    
    def on_epoch_end(self):
        """ Shuffle indices at end of epoch """
        if self.shuffle:
            np.random.shuffle(self.indices)


def create_tf_dataset(X, y, batch_size=32, augment=True, shuffle=True):
    """
    Create TensorFlow dataset with augmentation
    
    Args:
        X: Images
        y: Labels
        batch_size: Batch size
        augment: Whether to augment
        shuffle: Whether to shuffle
        
    Returns:
        tf.data.Dataset
    """

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # Shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(len(X), 1000))

    # Batch
    dataset = dataset.batch(batch_size)

    # Augmentation
    if augment:
        augmentation = create_augmentation_layer()
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls = tf.data.AUTOTUNE
        )

    # Prefetch for performance
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset