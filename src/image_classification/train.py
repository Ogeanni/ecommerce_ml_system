""" Training pipeline for image classification models """

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

class ImageClassificationTrainer:
    """ Train image classification models """
    def __init__(self, model, data, model_name="image_classifier"):
        """
        Initialize trainer
        
        Args:
            model: Keras model to train
            data: Dictionary with train/val/test data
            model_name: Name for saving models
        """
        self.model = model
        self.data = data
        self.model_name = model_name
        self.history = None
        
        # Create directories
        self.model_dir = Path(f'models/image_classification/saved_models')
        self.checkpoint_dir = Path(f'models/image_classification/checkpoints')
        self.log_dir = Path(f'models/image_classification/logs')

        for directory in [self.model_dir, self.checkpoint_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def compile_model(
            self,
            optimizer="adam",
            learning_rate=0.001,
            loss="sparse_categorical_crossentropy",
            metrics=None
    ):
        """
        Compile the model
        
        Args:
            optimizer: Optimizer name or instance
            learning_rate: Learning rate
            loss: Loss function
            metrics: List of metrics
        """
        print("\n" + "="*80)
        print("COMPILING MODEL")
        print("="*80)

        if metrics is None:
            metrics = ['accuracy']

        # Default metrics
        if optimizer == 'adam':
            opt = keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'sgd':
            opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        elif optimizer == 'rmsprop':
            opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            opt = optimizer

        # Compile
        self.model.compile(optimizer=opt, loss=loss, metrics=metrics)

        print(f"   Model compiled")
        print(f"   Optimizer: {optimizer}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Loss: {loss}")
        print(f"   Metrics: {[m if isinstance(m, str) else m.name for m in metrics]}")

    def create_callbacks(self,
                         early_stopping_patience=10,
                         reduce_lr_patience=5,
                         save_best_only=True):
        """
        Create training callbacks
         This function creates a set of training supervisors that:
           - Save the best model
           - Stop training when learning stops
           - Adjust learning rate automatically
           - Log training for inspection and debugging
        NB: They do NOT learn. They manage learning.
        
        Args:
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
            save_best_only: Whether to save only best model
            
        Returns:
            List of callbacks
        """
        callback_list = []

        # Model checkpoint
        checkpoint_path = self.checkpoint_dir / f"{self.model_name}_best.h5"
        checkpoint_callback = callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=save_best_only,
            save_weights_only=False,
            mode="max",
            verbose=1
        )
        callback_list.append(checkpoint_callback)

        # Early stopping
        early_stopping = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode='max',
            verbose=1
        )
        callback_list.append(early_stopping)

        # Reduce learning rate on plateau
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            mode='min',
            verbose=1
        )
        callback_list.append(reduce_lr)

        # TensorBoard
        log_path = self.log_dir / f'{self.model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        tensorboard_callback = callbacks.TensorBoard(
            log_dir=str(log_path),
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        )
        callback_list.append(tensorboard_callback)

        # CSV Logger
        csv_path = self.log_dir / f'{self.model_name}_training_log.csv'
        csv_logger = callbacks.CSVLogger(str(csv_path))
        callback_list.append(csv_logger)

        print(f"\n   Created {len(callback_list)} callbacks:")
        print(f"   - ModelCheckpoint: {checkpoint_path}")
        print(f"   - EarlyStopping (patience={early_stopping_patience})")
        print(f"   - ReduceLROnPlateau (patience={reduce_lr_patience})")
        print(f"   - TensorBoard: {log_path}")
        print(f"   - CSVLogger: {csv_path}")

        return callback_list
    
    def train(self,
              epochs=50,
              batch_size=32,
              validation_data=None,
              callbacks_list=None,
              class_weight=None,
              verbose=1):
        """
        Train the model
        
        Args:
            epochs: Number of epochs
            batch_size: Batch size
            validation_data: Validation data tuple (X_val, y_val)
            callbacks_list: List of callbacks
            class_weight: Class weights for imbalanced data
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        print("\n" + "="*80)
        print("TRAINING MODEL")
        print("="*80)

        # Prepare validation data
        if validation_data is None:
            validation_data = (self.data['X_val'], self.data['y_val'])

        # Calculate class weights if needed
        if class_weight == 'balanced':
            class_weight = self._calculate_class_weights()
            print(f"\nUsing balanced class weights:")
            for cls, weight in class_weight.items():
                print(f"  Class {cls}: {weight:.4f}")

        print(f"\nTraining parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training samples: {len(self.data['X_train']):,}")
        print(f"  Validation samples: {len(self.data['X_val']):,}")

        # Train
        self.history = self.model.fit(
            self.data['X_train'],
            self.data['y_train'],
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks_list,
            class_weight=class_weight,
            verbose=verbose
        )
        
        print(f"\n Training complete!")
        
        return self.history
    

    def _calculate_class_weights(self):
        """ Calculate class weights for imbalanced data """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(self.data['y_train'])
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=self.data['y_train']
        )
        
        return dict(zip(classes, weights))
    

    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save plot
        """
        if self.history is None:
            print("No training history available")
            return
        
        history = self.history.history
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        axes[0].plot(history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[1].plot(history['loss'], label='Train Loss', linewidth=2)
        axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Model Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n Training history plot saved to {save_path}")
        
        plt.show()
        plt.close()
    

    def save_model(self, save_path=None):
        """ Save trained model
        
        Args:
            save_path: Path to save model 
        """
        if save_path is None:
            save_path = self.model_dir / f'{self.model_name}_final.h5'
        
        self.model.save(save_path)
        print(f"\n Model saved to {save_path}")
        
        # Save training history
        if self.history:
            history_path = Path(save_path).parent / f'{self.model_name}_history.json'
            with open(history_path, 'w') as f:
                # Convert numpy values to python types
                history_dict = {
                    key: [float(val) for val in values] 
                    for key, values in self.history.history.items()
                }
                json.dump(history_dict, f, indent=2)
            print(f" Training history saved to {history_path}")

    
    def fine_tune(self, base_model_layers_to_unfreeze=20, epochs=10, learning_rate=1e-5):
        """
        Fine-tune a transfer learning model
        
        Args:
            base_model_layers_to_unfreeze: Number of base model layers to unfreeze
            epochs: Number of fine-tuning epochs
            learning_rate: Lower learning rate for fine-tuning
        """
        print("\n" + "="*80)
        print("FINE-TUNING MODEL")
        print("="*80)
        
        # Unfreeze layers
        for layer in self.model.layers:
            if hasattr(layer, 'layers'):  # This is the base model
                # This temporarily unfreezes everything inside the base model
                layer.trainable = True
                # Freeze all except last N layers
                for sublayer in layer.layers[:-base_model_layers_to_unfreeze]:
                    sublayer.trainable = False
        
        print(f"\nUnfrozen last {base_model_layers_to_unfreeze} layers of base model")
        
        # Recompile with lower learning rate
        self.compile_model(learning_rate=learning_rate)
        
        # Create new callbacks for fine-tuning
        callbacks_list = self.create_callbacks(
            early_stopping_patience=5,
            reduce_lr_patience=3
        )
        
        # Continue training
        print(f"\nFine-tuning for {epochs} epochs with lr={learning_rate}")

        # Detect whether X_train is a tf.data.Dataset
        X_train = self.data['X_train']
        y_train = self.data['y_train']
        X_val = self.data['X_val']
        y_val = self.data['y_val']

        if isinstance(X_train, tf.data.Dataset):
            # Dataset already contains both features and labels
            history_finetune = self.model.fit(
            X_train,
            epochs=epochs,
            validation_data=X_val,
            callbacks=callbacks_list,
            verbose=1
        )
        else:
            history_finetune = self.model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Combine histories
        if self.history:
            for key in history_finetune.history.keys():
                self.history.history[key].extend(history_finetune.history[key])
        else:
            self.history = history_finetune
        
        print(f"\n Fine-tuning complete!")

