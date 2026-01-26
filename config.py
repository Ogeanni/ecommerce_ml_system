from pathlib import Path
import sys
import os

# Detect if running on Colab
ON_COLAB = "COLAB_GPU" in os.environ or "google.colab" in sys.modules

if ON_COLAB:
    ROOT_DIR = Path("/content/ecommerce_ml_system")  # After cloning repo in Colab
else:
    ROOT_DIR = Path(__file__).resolve().parent.parent  # Local path to repo root

DATA_DIR = ROOT_DIR / "data"

# Add ROOT_DIR to sys.path for imports
sys.path.append(str(ROOT_DIR))
