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
    'model_type': 'simple',                     # 'simple', 'transfer', 'residual'
    'base_model_name': None,                    # For transfer learning
    
    # Training
    'batch_size': 32,
    'epochs': 20,
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

trainer.save_model()

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