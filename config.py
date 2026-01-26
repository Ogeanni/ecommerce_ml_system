from pathlib import Path
import sys
import os

# Detect if running in Colab
if "google.colab" in sys.modules:
    ROOT_DIR = Path("/content/drive/MyDrive/ecommerce_ml_system")
else:
    ROOT_DIR = Path(__file__).resolve().parent

sys.path.append(str(ROOT_DIR))

# Output directories
RESULTS_DIR = ROOT_DIR / "results/image_classification"
MODELS_DIR = ROOT_DIR / "models/image_classification"
LOGS_DIR = ROOT_DIR / "logs/image_classification"

# Make sure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Create folders if missing
for d in [RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)