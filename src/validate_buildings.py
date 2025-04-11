"""
Building-level validation using information fusion.

This script evaluates a trained model on building-level data, using multiple
images per building and applying information fusion to make robust predictions.
"""

import os
import argparse
import logging
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report, confusion_matrix

# Import from other modules
from fusion import InformationFusion, optimal_decision

# Import configuration
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, PREDICTIONS_DIR,
    IMG_WIDTH, IMG_HEIGHT, DECISION_COST
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_preprocessing(model_path):
    """
    Load a saved model and determine its preprocessing function.

    Parameters
    ----------
    model_path : str
        Path to the saved model file

    Returns
    -------
    model : tensorflow.keras.models.Model
        The loaded model
    preprocess_input : function
        Preprocessing function for the model
    """
    # Load the model
    model = load_model(model_path)

    # Determine preprocessing function based on model name
    if 'xception' in model_path.lower():
        from tensorflow.keras.applications.xception import preprocess_input
    elif 'inception_v3' in model_path.lower():
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    elif 'inception_resnet' in model_path.lower():
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    else:
        # Default to Xception preprocessing if unsure
        logger.warning("Could not determine model type from filename. Using Xception preprocessing.")
        from tensorflow.keras.applications.xception import preprocess_input

    return model, preprocess_input


def extract_building_info(building_dir):
    """
    Extract building ID and true class from building directory name.

    The directory name format is expected to be 'building_id_class'.

    Parameters
    ----------
    building_dir : str
        Building directory path

    Returns
    -------
    tuple
        (building_id, true_class)
    """
    dir_name = os.path.basename(building_dir)

    # If the directory name contains an underscore, split by the last one
    if '_' in dir_name:
        # Find the last underscore position
        last_underscore = dir_name.rindex('_')
        building_id = dir_name[:last_underscore]
        true_class = dir_name[last_underscore + 1:]
    else:
        # If no underscore, use the whole name as building_id and set true_class to None
        building_id = dir_name
        true_class = None

    return building_id, true_class


def predict_building_damage(model, preprocess_input, building_dir, visualization_dir=None):
    """
    Predict damage for a building using multiple images and information fusion.

    Parameters
    ----------
    model : tensorflow.keras.models.Model
        Trained model
    preprocess_input : function
        Preprocessing function for the model
    building_dir : str
        Directory containing building images
    visualization_dir : str, optional
        Directory to save visualizations. If None, no visualizations are saved.

    Returns
    -------
    dict
        Dictionary containing prediction results
    """
    # Get building ID and true class
    building_id, true_class = extract_building_info(building_dir)

    # Get list of images for this building
    image_paths = glob.glob(os.path.join(building_dir, "*.jpg")) + \
                  glob.glob(os.path.join(building_dir, "*.jpeg")) + \
                  glob.glob(os.path.join(building_dir, "*.png"))

    if not image_paths:
        logger.warning(f"No images found in {building_dir}")
        return {
            'building_id': building_id,
            'true_class': true_class,
            'num_images': 0,
            'damage_probabilities': [],
            'fused_probability': None,
            'decision': 'no_decision',
            'visualization_path': None
        }

    # Initialize list to store damage probabilities
    damage_probabilities = []
    annotated_images = []

    # Process each image
    for img_path in image_paths:
        # Load and preprocess image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_image = image.copy()

        # Resize and preprocess
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = preprocess_input(image.astype(np.float32))
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image, verbose=0)[0]
        damage_probability = prediction[1]  # Assuming binary classification with damage as index 1
        damage_probabilities.append(damage_probability)

        # Create annotated image if visualization is requested
        if visualization_dir:
            # Create annotation
            annotated = orig_image.copy()
            h, w = annotated.shape[:2]

            # Add prediction text
            text = f"Damage prob: {damage_probability:.2f}"
            cv2.putText(annotated, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0) if damage_probability > 0.5 else (255, 0, 0), 2)

            annotated_images.append(annotated)

    # Apply information fusion
    fuser = InformationFusion(lambda alpha: np.ceil(np.mean(alpha)))
    fused_probability = fuser(damage_probabilities)

    # Make decision based on fused probability
    decision = optimal_decision(fused_probability, DECISION_COST)

    # Create visualization if requested
    visualization_path = None
    if visualization_dir and annotated_images:
        # Create a grid of images
        n_images = len(annotated_images)
        cols = min(5, n_images)
        rows = (n_images + cols - 1) // cols

        # Check max dimensions
        max_h = max(img.shape[0] for img in annotated_images)
        max_w = max(img.shape[1] for img in annotated_images)

        # Resize all images to the same size
        annotated_images = [cv2.resize(img, (max_w, max_h)) for img in annotated_images]

        # Create grid
        grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)

        for i, img in enumerate(annotated_images):
            row = i // cols
            col = i % cols
            grid[row * max_h:(row + 1) * max_h, col * max_w:(col + 1) * max_w] = img

        # Add decision information at the bottom
        info_height = 100
        info_img = np.ones((info_height, grid.shape[1], 3), dtype=np.uint8) * 255

        # Add text with decision
        cv2.putText(info_img, f"Building ID: {building_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        if true_class:
            cv2.putText(info_img, f"True class: {true_class}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        color = (0, 200, 0) if decision == true_class else (200, 0, 0)
        cv2.putText(info_img, f"Decision: {decision} (Prob: {fused_probability:.2f})",
                    (grid.shape[1] // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Combine grid and info
        final_vis = np.vstack([grid, info_img])

        # Save visualization
        os.makedirs(visualization_dir, exist_ok=True)
        vis_filename = f"{building_id}_{true_class}_{decision}.jpg"
        visualization_path = os.path.join(visualization_dir, vis_filename)
        cv2.imwrite(visualization_path, cv2.cvtColor(final_vis, cv2.COLOR_RGB2BGR))

    # Return results
    return {
        'building_id': building_id,
        'true_class': true_class,
        'num_images': len(image_paths),
        'damage_probabilities': damage_probabilities,
        'fused_probability': fused_probability,
        'decision': decision,
        'visualization_path': visualization_path
    }


def validate_buildings(args):
    """
    Validate model performance on buildings with multiple images.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model, preprocess_input = load_model_and_preprocessing(args.model_path)

    # Get buildings directory
    buildings_dir = args.buildings_dir or os.path.join(PROCESSED_DATA_DIR, 'validation_buildings')

    if not os.path.exists(buildings_dir):
        logger.error(f"Buildings directory {buildings_dir} does not exist!")
        return

    # Get list of building directories
    building_dirs = [d for d in glob.glob(os.path.join(buildings_dir, '*'))
                     if os.path.isdir(d)]

    logger.info(f"Found {len(building_dirs)} buildings to validate")

    # Setup visualization directory
    visualization_dir = None
    if args.visualize:
        model_name = os.path.basename(args.model_path).split('.')[0]
        visualization_dir = os.path.join(PREDICTIONS_DIR, 'building_visualizations', model_name)
        os.makedirs(visualization_dir, exist_ok=True)

    # Process each building
    results = []
    for building_dir in building_dirs:
        logger.info(f"Processing building: {os.path.basename(building_dir)}")

        # Predict damage for this building
        result = predict_building_damage(
            model,
            preprocess_input,
            building_dir,
            visualization_dir if args.visualize else None
        )

        results.append(result)

    # Filter results to only include buildings with known true class
    valid_results = [r for r in results if r['true_class'] is not None]

    # Calculate metrics if we have valid results
    if valid_results:
        # Get true and predicted classes
        y_true = [r['true_class'] for r in valid_results]
        y_pred = [r['decision'] for r in valid_results]

        # Remove 'no_decision' entries for metrics calculation
        valid_indices = [i for i, pred in enumerate(y_pred) if pred != 'no_decision']
        y_true_valid = [y_true[i] for i in valid_indices]
        y_pred_valid = [y_pred[i] for i in valid_indices]

        # Calculate metrics if we have valid predictions
        if y_true_valid and y_pred_valid:
            # Get unique classes
            classes = sorted(list(set(y_true_valid) | set(y_pred_valid)))

            # Generate classification report
            report = classification_report(y_true_valid, y_pred_valid,
                                           target_names=classes,
                                           output_dict=True)

            logger.info(f"Building-level classification report:\n{pd.DataFrame(report).transpose()}")

            # Calculate confusion matrix
            cm = confusion_matrix(y_true_valid, y_pred_valid, labels=classes)

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Building-level Confusion Matrix')
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            # Add text annotations to the confusion matrix
            thresh = cm.max() / 2
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                             ha="center", va="center",
                             color="white" if cm[i, j] > thresh else "black")

            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

            # Save the figure
            os.makedirs(RESULTS_DIR, exist_ok=True)
            plt.savefig(os.path.join(RESULTS_DIR, 'building_confusion_matrix.png'), dpi=300)

        # Count decision types
        decision_counts = {}
        for result in results:
            decision = result['decision']
            decision_counts[decision] = decision_counts.get(decision, 0) + 1

        logger.info(f"Decision counts: {decision_counts}")

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_path = os.path.join(RESULTS_DIR, 'building_validation_results.csv')
    # Convert lists to strings for CSV storage
    results_df['damage_probabilities'] = results_df['damage_probabilities'].apply(lambda x: str(x))
    results_df.to_csv(results_path, index=False)

    logger.info(f"Results saved to {results_path}")


def main():
    """
    Main function to parse arguments and start validation.
    """
    parser = argparse.ArgumentParser(
        description='Validate a building damage classification model on building-level data')

    # Model parameters
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model file (.h5)')

    # Validation parameters
    parser.add_argument('--buildings-dir', type=str, default=None,
                        help='Directory containing building subdirectories with images')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations for building predictions')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.error(f"Model file {args.model_path} does not exist!")
        return

    # Validate the model on buildings
    validate_buildings(args)


if __name__ == "__main__":
    main()