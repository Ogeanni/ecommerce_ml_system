"""
Complete end-to-end training script for image classification
"""

import sys
from pathlib import Path

from config import ROOT_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR

import tensorflow as tf
import numpy as np

# Import custom modules
from src.image_classification.data_loader import load_image_data
from src.image_classification.data_loader_2 import load_image_data_2
from src.image_classification.augmentation import create_augmentation_layer
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
    'img_size': (124, 124),                     # Changed from (124, 124) to standard (224, 224)
    
    # Model
    'model_type': 'transfer',                     # 'simple', 'transfer', 'residual'
    'base_model_name': 'mobilenetv2',           #  Set default for transfer learning
    
    # Training
    'batch_size': 32,
    'epochs': 5,
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
    
    # Saving
    'save_format': 'tf',                        
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
    img_size=CONFIG['img_size'],
    batch_size=CONFIG['batch_size']
)

print(f"\n Data loaded successfully!")
print(f"  Classes: {data['class_names']}")
print(f"  Number of classes: {data['num_classes']}")

train_dataset = data["X_train"]
val_dataset   = data["X_val"]
test_dataset  = data["X_test"]

print(f"\nDataset types:")
print(f"  Train: {type(train_dataset)}")
print(f"  Val: {type(val_dataset)}")
print(f"  Test: {type(test_dataset)}")

# ==================== STEP 2: DATA AUGMENTATION ====================
print("\n" + "="*80)
print("STEP 2: DATA AUGMENTATION")
print("="*80)

if CONFIG['use_augmentation']:
    print("Creating augmentation layer...")
    augmentation_layer = create_augmentation_layer()

    # Apply augmentation only on training dataset
    train_dataset = train_dataset.map(
        lambda x, y: (augmentation_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    print(" Augmentation applied to training data")
else:
    print("  Skipping data augmentation")

# Prefetch for performance
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)  # FIX: Use AUTOTUNE instead of 1
val_dataset   = val_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset  = test_dataset.prefetch(tf.data.AUTOTUNE)

print(f"\n Datasets optimized with prefetching")

# ==================== STEP 3: BUILD MODEL ====================
print("\n" + "="*80)
print("STEP 3: BUILDING MODEL")
print("="*80)

if CONFIG["model_type"] == "transfer":
    if CONFIG["base_model_name"] is None:
        raise ValueError("base_model_name must be specified for transfer learning")
    
    model = build_model(
        model_type="transfer",
        input_shape=(*CONFIG["img_size"], 3),
        num_classes=data["num_classes"],
        base_model_name=CONFIG["base_model_name"],
        trainable_layers=0  # Freeze base model initially
    )
else:
    model = build_model(
        model_type=CONFIG["model_type"],
        input_shape=(*CONFIG["img_size"], 3),
        num_classes=data["num_classes"]
    )
    
print("\n Model built successfully")
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
    reduce_lr_patience=CONFIG['reduce_lr_patience'],
    save_format=CONFIG['save_format']  # FIX: Pass save_format
)

# ==================== STEP 5: TRAIN MODEL ====================
print("\n" + "="*80)
print("STEP 5: TRAINING MODEL")
print("="*80)

# FIX: Always use the trainer.train() method for consistency
history = trainer.train(
    epochs=CONFIG['epochs'],
    batch_size=CONFIG['batch_size'],  # Will be ignored for tf.data.Dataset
    callbacks_list=callbacks_list
)

print("\n Training complete!")

# Plot training history
trainer.plot_training_history(
    save_path=RESULTS_DIR / 'training_history.png'
)

# ==================== STEP 6: FINE-TUNING (Optional) ====================
# if CONFIG['model_type'] == 'transfer' and CONFIG['fine_tune']:
#     print("\n" + "="*80)
#     print("STEP 6: FINE-TUNING MODEL")
#     print("="*80)
    
#     trainer.fine_tune(
#         base_model_layers_to_unfreeze=20,
#         epochs=CONFIG['fine_tune_epochs'],
#         learning_rate=CONFIG['fine_tune_lr']
#     )
    
#     print("\n Fine-tuning complete!")
    
#     # Plot updated training history
#     trainer.plot_training_history(
#         save_path=RESULTS_DIR / 'training_history_with_finetune.png'
#     )

# ==================== STEP 7: SAVE MODEL ====================
print("\n" + "="*80)
print("STEP 7: SAVING MODEL")
print("="*80)

trainer.save_model(save_format=CONFIG['save_format'])

print(f"\n Model saved in {CONFIG['save_format']} format")

# ==================== STEP 8: EVALUATE MODEL ====================
# print("\n" + "="*80)
# print("STEP 8: EVALUATING MODEL")
# print("="*80)

# evaluator = ImageClassificationEvaluator(
#     model=model,
#     data=data,
#     class_names=data['class_names']
# )

# # Generate comprehensive evaluation report
# evaluator.generate_full_report(
#     dataset='test',
#     output_dir=RESULTS_DIR
# )

# print(f"\n Evaluation complete! Results saved to {RESULTS_DIR}")

# ==================== STEP 9: FINAL SUMMARY ====================
# print("\n" + "="*80)
# print("TRAINING PIPELINE COMPLETE!")
# print("="*80)

# # Evaluate on all datasets
# print("\nFinal Performance Summary:")
# print("-" * 80)

# for dataset in ['train', 'val', 'test']:
#     metrics = evaluator.evaluate(dataset=dataset, verbose=0)
#     print(f"\n{dataset.upper()} SET:")
#     for metric_name, metric_value in metrics.items():
#         print(f"  {metric_name}: {metric_value:.4f}")

# print("\n" + "="*80)
# print("FILES SAVED:")
# print("="*80)
# print(f"  Model: {MODELS_DIR / 'saved_models'}")
# print(f"  Checkpoints: {MODELS_DIR / 'checkpoints'}")
# print(f"  Logs: {LOGS_DIR}")
# print(f"  Results: {RESULTS_DIR}")
# print("\n All done!")