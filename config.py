# from pathlib import Path
# import sys
# import os

# # Check if running in Colab
# if os.path.exists('/content/drive'):
#     # Running in Colab with Drive mounted
#     ROOT_DIR = Path('/content/drive/MyDrive/ecommerce_ml_system')
# else:
#     # Running locally
#     ROOT_DIR = Path(__file__).resolve().parent[1]

# sys.path.append(str(ROOT_DIR))

# # Output directories
# RESULTS_DIR = ROOT_DIR / "results/image_classification"
# MODELS_DIR = ROOT_DIR / "models/image_classification"
# LOGS_DIR = ROOT_DIR / "logs/image_classification"

# # Make sure directories exist
# RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# MODELS_DIR.mkdir(parents=True, exist_ok=True)
# LOGS_DIR.mkdir(parents=True, exist_ok=True)


# # Data directories
# DATA_DIR = ROOT_DIR / 'data'
# RAW_DATA_DIR = DATA_DIR / 'raw'
# PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# # Model directories
# MODELS_DIR = ROOT_DIR / 'models' / 'image_classification'
# SAVED_MODELS_DIR = MODELS_DIR / 'saved_models'
# CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'

# # Logs and results
# LOGS_DIR = ROOT_DIR / 'models' / 'image_classification' / 'logs'
# RESULTS_DIR = ROOT_DIR / 'results' / 'image_classification'

# # Create directories if they don't exist
# for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
#                   MODELS_DIR, SAVED_MODELS_DIR, CHECKPOINTS_DIR, 
#                   LOGS_DIR, RESULTS_DIR]:
#     directory.mkdir(parents=True, exist_ok=True)
    
# print(f" Root directory: {ROOT_DIR}")
# print(f" Models will be saved to: {SAVED_MODELS_DIR}")







from pathlib import Path
import os
import sys

# ===============================
# Detect environment (Colab vs Local)
# ===============================
if os.path.exists('/content/drive'):
    # Running in Colab with Drive mounted
    ROOT_DIR = Path('/content/drive/MyDrive/ecommerce_ml_system')
else:
    # Running locally
    ROOT_DIR = Path(__file__).resolve().parent

sys.path.append(str(ROOT_DIR))

# ===============================
# Base directories
# ===============================
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
LOGS_DIR = ROOT_DIR / "logs"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ===============================
# Task-specific directory factory
# ===============================
def get_task_dirs(task_name: str):
    """
    task_name examples:
    - image_classification
    - object_detection
    - regression
    - tabular_classification
    """

    task_dirs = {
        "models": MODELS_DIR / task_name,
        "saved_models": MODELS_DIR / task_name / "saved_models",
        "checkpoints": MODELS_DIR / task_name / "checkpoints",
        "preprocessing": MODELS_DIR / task_name / "preprocessing",
        "logs": LOGS_DIR / task_name,
        "results": RESULTS_DIR / task_name,
    }

    for d in task_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return task_dirs


# ===============================
# Create base directories
# ===============================
for d in [
    DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
    MODELS_DIR, RESULTS_DIR, LOGS_DIR
]:
    d.mkdir(parents=True, exist_ok=True)


print(f"ROOT_DIR: {ROOT_DIR}")
