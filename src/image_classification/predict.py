"""
Inference pipeline for image classification
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2
from pathlib import Path
import json
from typing import Union, List, Tuple
import matplotlib.pyplot as plt
