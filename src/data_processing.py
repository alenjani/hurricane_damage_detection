"""
Data processing utilities for the building damage classifier.

This module provides functions for:
1. Splitting raw data into training and validation sets
2. Processing and preparing images for model training
"""

import os
import shutil
import random
import logging
from pathlib import Path
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from collections import defaultdict

# Import configuration
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, CLASS_MAPPING, REV_CLASS_MAPPING

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_building_id(filename):
    """
    Extract building ID from a filename.

    The expected format is 'building_id__image_id.ext'

    Parameters
    ----------
    filename : str
        Filename to parse

    Returns
    -------
    str
        The extracted building ID or the filename without extension if no
        separator is found
    """
    basename = Path(filename).stem
    if '__' in basename:
        return basename.split('__')[0]
    return basename


def prepare_directory_structure():
    """
    Create necessary directory structure for processed data.
    """
    # Create processed data directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # Create train and validation directories
    for split in ['train', 'val']:
        split_dir = os.path.join(PROCESSED_DATA_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

        # Create class directories
        for class_name in CLASS_MAPPING.keys():
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

    logger.info("Directory structure prepared.")


def split_data(train_ratio=0.7, random_seed=42):
    """
    Split raw data into training and validation sets by building ID.

    Parameters
    ----------
    train_ratio : float, optional
        Ratio of data to use for training, by default 0.7
    random_seed : int, optional
        Random seed for reproducibility, by default 42
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    # First, prepare directory structure
    prepare_directory_structure()

    # Get class mapping for directory names
    for class_name, original_class in CLASS_MAPPING.items():
        source_dir = os.path.join(RAW_DATA_DIR, class_name)

        if not os.path.exists(source_dir):
            logger.warning(f"Source directory {source_dir} does not exist! Skipping.")
            continue

        # Group files by building ID
        building_files = defaultdict(list)
        for file in os.listdir(source_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                building_id = extract_building_id(file)
                building_files[building_id].append(file)

        # Get all unique building IDs
        building_ids = list(building_files.keys())
        random.shuffle(building_ids)

        # Split building IDs into train and validation sets
        train_size = int(len(building_ids) * train_ratio)
        train_building_ids = building_ids[:train_size]
        val_building_ids = building_ids[train_size:]

        logger.info(f"Class {class_name}: {len(train_building_ids)} buildings for training, "
                    f"{len(val_building_ids)} buildings for validation")

        # Copy files to appropriate directories
        for split, building_set in [('train', train_building_ids), ('val', val_building_ids)]:
            target_dir = os.path.join(PROCESSED_DATA_DIR, split, class_name)

            for building_id in building_set:
                for file in building_files[building_id]:
                    source_path = os.path.join(source_dir, file)
                    target_path = os.path.join(target_dir, file)
                    shutil.copy2(source_path, target_path)

        # Count files
        train_count = len([f for bid in train_building_ids for f in building_files[bid]])
        val_count = len([f for bid in val_building_ids for f in building_files[bid]])

        logger.info(f"Class {class_name}: {train_count} images for training, "
                    f"{val_count} images for validation")

    logger.info("Data splitting complete.")


def create_validation_building_structure():
    """
    Create a structure for validation by building.

    This function organizes validation images by building ID to enable
    building-level evaluation and information fusion.
    """
    val_dir = os.path.join(PROCESSED_DATA_DIR, 'val')
    building_val_dir = os.path.join(PROCESSED_DATA_DIR, 'validation_buildings')

    # Create validation buildings directory
    os.makedirs(building_val_dir, exist_ok=True)

    # Collect all building IDs from validation set
    building_class_map = {}
    for class_name in CLASS_MAPPING.keys():
        class_dir = os.path.join(val_dir, class_name)

        if not os.path.exists(class_dir):
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                building_id = extract_building_id(file)
                building_class_map[building_id] = class_name

    # Create a directory for each building and copy its images
    for class_name in CLASS_MAPPING.keys():
        class_dir = os.path.join(val_dir, class_name)

        if not os.path.exists(class_dir):
            continue

        for file in os.listdir(class_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                building_id = extract_building_id(file)
                building_dir = os.path.join(building_val_dir, f"{building_id}_{class_name}")

                # Create building directory if it doesn't exist
                os.makedirs(building_dir, exist_ok=True)

                # Copy image to building directory
                source_path = os.path.join(class_dir, file)
                target_path = os.path.join(building_dir, file)
                shutil.copy2(source_path, target_path)

    logger.info(f"Created validation building structure with {len(building_class_map)} buildings.")


def check_dataset_balance():
    """
    Check and report on dataset balance.
    """
    class_counts = {}
    for split in ['train', 'val']:
        class_counts[split] = {}
        for class_name in CLASS_MAPPING.keys():
            class_dir = os.path.join(PROCESSED_DATA_DIR, split, class_name)
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                class_counts[split][class_name] = count
            else:
                class_counts[split][class_name] = 0

    # Print report
    logger.info("Dataset balance report:")
    for split in ['train', 'val']:
        total = sum(class_counts[split].values())
        logger.info(f"  {split} set: {total} total images")
        for class_name, count in class_counts[split].items():
            percentage = (count / total * 100) if total > 0 else 0
            logger.info(f"    {class_name}: {count} images ({percentage:.1f}%)")


def main():
    """
    Main function to execute data processing steps.
    """
    parser = argparse.ArgumentParser(description='Process building damage classification dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of data to use for training (default: 0.7)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()

    logger.info("Starting data processing...")

    # Check if raw data exists
    if not os.path.exists(RAW_DATA_DIR):
        logger.error(f"Raw data directory {RAW_DATA_DIR} does not exist! Please create it and add data.")
        return

    # Check if required class directories exist
    for class_name in CLASS_MAPPING.keys():
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            logger.error(f"Class directory {class_dir} does not exist! Please check your data structure.")
            return

    # Split data into training and validation sets
    split_data(train_ratio=args.train_ratio, random_seed=args.seed)

    # Create validation building structure
    create_validation_building_structure()

    # Check dataset balance
    check_dataset_balance()

    logger.info("Data processing complete.")


if __name__ == "__main__":
    main()