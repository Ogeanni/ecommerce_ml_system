"""
Build CNN models for image classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    ResNet50, MobileNetV2, EfficientNetB0, VGG16
)


class ImageClassificationModelBuilder:
    """ Build various CNN Models """
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        """
        Initialize model builder
        
        Args:
            input_shape: Input image shape
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_simple_cnn(self):
        """
        Build simple CNN from scratch
        
        Returns:
            Keras model
        """

        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.4),

            # Dense layers
            layers.Flatten(),                                           
            layers.Dense(256, activation="relu"),                       
            layers.Dropout(0.5),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation="softmax")       
            ], name="simple_cnn")

        return model
    
    def build_transfer_learning_model(self, base_model_name="mobilenetv2", trainable_layers=0):
        """
        Build model using transfer learning
        
        Args:
            base_model_name: Name of pretrained model
            trainable_layers: Number of base model layers to make trainable
            
        Returns:
            Keras model
        """
        # Load pretrained model
        if base_model_name == "resnet50":
            base_model = ResNet50(weights="imagenet", include_top=False, input_shape=self.input_shape)
        elif base_model_name == "mobilenetv2":
            base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=self.input_shape)
        elif base_model_name == "efficientnetb0":
            base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=self.input_shape)
        elif base_model_name == "vgg16":
            base_model = VGG16(weights="imagenet", include_top=False, input_shape=self.input_shape)
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze base model. 
        base_model.trainable = False

        # Optionally unfreeze last few layers
        if trainable_layers > 0:
            for layer in base_model.layers[-trainable_layers:]:
                layer.trainable = True

        # Build full model
        inputs = keras.Input(shape=self.input_shape)

        # Preprocessing for pretrained models
        if base_model_name == "mobilenetv2":
            x = keras.applications.mobilenet_v2.preprocess_input(inputs)
        elif base_model_name == "resnet50":
            x = keras.applications.resnet50.preprocess_input(inputs)
        elif base_model_name == "vgg16":
            x = keras.applications.vgg16.preprocess_input(inputs)
        elif base_model_name == "efficientnetb0":
            x = keras.applications.efficientnet.preprocess_input(inputs)
        else:
            x = inputs

        # Base model
        x = base_model(x, training=False)

        # Classification head (custom). 
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs, name=f"transfer_{base_model_name}")

        return model
    
    def build_residual_cnn(self):
        """
        Build custom CNN with residual connections
        
        Returns:
            Keras model
        """
        def residual_block(x, filters, kernel_size=3):
            """ Residual block """
            shortcut = x

            # Conv layers
            x = layers.Conv2D(filters, kernel_size, padding="same")(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(filters, kernel_size, padding="same")(x)
            x = layers.BatchNormalization()(x)

            # Add shortcut
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)

            x = layers.Add()([x, shortcut])
            x = layers.Activation("relu")(x)

            return x
        # Input
        inputs = keras.Input(shape=self.input_shape)

        # Initial conv
        x = layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Residual blocks
        x = residual_block(x, 64)
        x = residual_block(x, 64)
        
        x = layers.MaxPooling2D(2)(x)
        x = residual_block(x, 128)
        x = residual_block(x, 128)
        
        x = layers.MaxPooling2D(2)(x)
        x = residual_block(x, 256)
        x = residual_block(x, 256)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs, name="residual_cnn")
        
        return model
    
def build_model(model_type="simple", input_shape=(224, 224, 3), num_classes=10, **kwargs):
    """
    Convenience function to build models
    
    Args:
        model_type: Type of model to build
        input_shape: Input shape
        num_classes: Number of classes
        **kwargs: Additional arguments for specific models
        
    Returns:
        Keras model
    """

    builder = ImageClassificationModelBuilder(input_shape, num_classes)

    if model_type == "simple":
        model = builder.build_simple_cnn()
    elif model_type == "transfer":
        model = builder.build_transfer_learning_model(**kwargs)
    elif model_type == "residual":
        model = builder.build_residual_cnn()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == "__main__":
    # Test model building
    print("Building Simple CNN...")
    model_simple = build_model("simple", num_classes=10)
    model_simple.summary()
    
    print("\n" + "="*80 + "\n")
    print("Building Transfer Learning Model (MobileNetV2)...")
    model_transfer = build_model("transfer", num_classes=10, base_model_name="mobilenetv2")
    model_transfer.summary()

