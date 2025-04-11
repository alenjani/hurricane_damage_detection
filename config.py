"""
Configuration file for the building damage classifier.
"""
import os
from pathlib import Path

# Directory setup
ROOT_DIR = Path(__file__).parent.absolute()
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
PREDICTIONS_DIR = os.path.join(RESULTS_DIR, 'predictions')

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
                 MODELS_DIR, RESULTS_DIR, FIGURES_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Class mapping
CLASS_MAPPING = {
    'major_damage': 'unusable',  # Map to original code naming
    'non_major_damage': 'usable'  # Map to original code naming
}

# Reversed class mapping
REV_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# Model training parameters
IMG_WIDTH = 299
IMG_HEIGHT = 299
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.000005
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-6

# Information fusion parameters
FUSION_THRESHOLD = 0.5  # Threshold for final decision

# Decision parameters
DECISION_COST = {
    'major_damage': {
        'major_damage': 0,
        'non_major_damage': 1,
        'no_decision': 0.3
    },
    'non_major_damage': {
        'major_damage': 1,
        'non_major_damage': 0,
        'no_decision': 0.3
    }
}