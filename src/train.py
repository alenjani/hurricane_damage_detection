"""
Training script for building damage classification models.

This script trains CNN models to classify building images as having
major damage or non-major damage after natural disasters.
"""

import os
import argparse
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import time

# Import from other modules
from models import create_model, get_class_weights, summary_to_dict

# Import configuration
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, RESULTS_DIR,
    IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, EPOCHS
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(args):
    """
    Train a CNN model for building damage classification.

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments
    """
    # Create model
    model, preprocess_input = create_model(
        model_name=args.model,
        num_classes=args.num_classes,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
    )

    # Show model summary
    model.summary()

    # Paths for training data
    train_data_dir = os.path.join(PROCESSED_DATA_DIR, 'train')
    validation_data_dir = os.path.join(PROCESSED_DATA_DIR, 'val')

    # Check if directories exist
    if not os.path.exists(train_data_dir) or not os.path.exists(validation_data_dir):
        logger.error("Training or validation directory doesn't exist. Run data_processing.py first.")
        return

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest'
    )

    # No augmentation for validation data, just preprocessing
    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Calculate class weights if enabled
    if args.use_class_weights:
        class_weight = get_class_weights(train_generator.classes)
        logger.info(f"Using class weights: {class_weight}")
    else:
        class_weight = None

    # Create output directories if they don't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Model file path
    model_file = os.path.join(MODELS_DIR, f"damage_classifier_{args.model}.h5")
    history_file = os.path.join(RESULTS_DIR, f"history_{args.model}.pickle")

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            model_file,
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=5,
            verbose=1,
            min_lr=1e-7
        )
    ]

    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // args.batch_size
    validation_steps = validation_generator.samples // args.batch_size

    # Ensure at least one step per epoch
    steps_per_epoch = max(steps_per_epoch, 1)
    validation_steps = max(validation_steps, 1)

    logger.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1
    )

    end_time = time.time()
    training_time = end_time - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Save training history
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)

    # Save model summary
    summary = summary_to_dict(model)
    summary['training_time'] = training_time
    summary['training_parameters'] = vars(args)

    summary_file = os.path.join(RESULTS_DIR, f"summary_{args.model}.pickle")
    with open(summary_file, 'wb') as file:
        pickle.dump(summary, file)

    # Plot training history
    plot_training_history(history, args.model)

    logger.info(f"Model saved to {model_file}")
    logger.info(f"Training history saved to {history_file}")


def plot_training_history(history, model_name):
    """
    Plot and save training history figures.

    Parameters
    ----------
    history : tensorflow.keras.callbacks.History
        Training history
    model_name : str
        Name of the model
    """
    # Create figure directory if it doesn't exist
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], 'b-', label='Training')
    plt.plot(history.history['val_accuracy'], 'r-', label='Validation')
    plt.title(f'Model Accuracy ({model_name})')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f"accuracy_{model_name}.png"), dpi=300)

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], 'b-', label='Training')
    plt.plot(history.history['val_loss'], 'r-', label='Validation')
    plt.title(f'Model Loss ({model_name})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(os.path.join(FIGURES_DIR, f"loss_{model_name}.png"), dpi=300)


def main():
    """
    Main function to parse arguments and start training.
    """
    parser = argparse.ArgumentParser(description='Train a building damage classification model')

    # Model parameters
    parser.add_argument('--model', type=str, default='xception',
                        choices=['xception', 'inception_v3', 'inception_resnet_v2'],
                        help='Model architecture to use (default: xception)')
    parser.add_argument('--num-classes', type=int, default=2,
                        help='Number of classes (default: 2)')

    # Training parameters
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                        help=f'Batch size (default: {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help=f'Number of epochs (default: {EPOCHS})')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use class weights to handle imbalanced data')

    args = parser.parse_args()

    # Train the model
    train_model(args)


if __name__ == "__main__":
    main()