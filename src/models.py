"""
CNN Model definitions for building damage classification.

This module provides functions to create and configure different CNN architectures
for image classification of building damage.
"""

import os
import logging
from tensorflow.keras import applications
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
import collections

# Import configuration
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_WIDTH, IMG_HEIGHT, LEARNING_RATE, WEIGHT_DECAY, MOMENTUM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_model(model_name="xception", num_classes=2, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)):
    """
    Create a CNN model for image classification.

    Parameters
    ----------
    model_name : str, optional
        Name of the model architecture to use, by default "xception"
        Options: "xception", "inception_v3", "inception_resnet_v2"
    num_classes : int, optional
        Number of output classes, by default 2
    input_shape : tuple, optional
        Shape of input images, by default (IMG_WIDTH, IMG_HEIGHT, 3)

    Returns
    -------
    tensorflow.keras.models.Model
        The configured model
    """
    logger.info(f"Creating {model_name} model with {num_classes} classes")

    # Select base model based on architecture name
    if model_name.lower() == "xception":
        base_model = applications.Xception(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )
        preprocess_input = applications.xception.preprocess_input

    elif model_name.lower() == "inception_v3":
        base_model = applications.InceptionV3(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )
        preprocess_input = applications.inception_v3.preprocess_input

    elif model_name.lower() == "inception_resnet_v2":
        base_model = applications.InceptionResNetV2(
            input_shape=input_shape,
            weights='imagenet',
            include_top=False
        )
        preprocess_input = applications.inception_resnet_v2.preprocess_input

    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    # Add classification layers on top of the base model
    x = Flatten(name='flatten')(base_model.output)
    x = Dense(1024, activation='relu', name='fc1')(x)
    x = Dense(1024, activation='relu', name='fc2')(x)
    predictions = Dense(num_classes, activation='softmax', name='predictions')(x)

    # Create the model
    model = Model(base_model.input, predictions)

    # Configure the model for training
    optimizer = SGD(
        learning_rate=LEARNING_RATE,
        momentum=MOMENTUM,
        decay=WEIGHT_DECAY,
        nesterov=True
    )

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=['accuracy']
    )

    return model, preprocess_input


def get_class_weights(class_counts, smooth_factor=0):
    """
    Calculate class weights for imbalanced datasets.

    Parameters
    ----------
    class_counts : list or array-like
        Number of samples in each class
    smooth_factor : float, optional
        Smoothing factor to apply, by default 0

    Returns
    -------
    dict
        Dictionary mapping class indices to weights
    """
    counter = collections.Counter(class_counts)

    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p

    majority = max(counter.values())

    return {cls: float(majority / count) for cls, count in counter.items()}


def summary_to_dict(model):
    """
    Convert a model's summary to a dictionary.

    Parameters
    ----------
    model : tensorflow.keras.models.Model
        The model to summarize

    Returns
    -------
    dict
        Dictionary containing model summary information
    """
    # Create a list to store the summary lines
    summary_list = []

    # Define a custom print function to capture summary
    def custom_print(line):
        summary_list.append(line)

    # Generate summary
    model.summary(print_fn=custom_print)

    # Process the summary list to extract key information
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for line in summary_list:
        if "Total params:" in line:
            total_params = int(line.split(":")[1].strip().replace(",", ""))
        elif "Trainable params:" in line:
            trainable_params = int(line.split(":")[1].strip().replace(",", ""))
        elif "Non-trainable params:" in line:
            non_trainable_params = int(line.split(":")[1].strip().replace(",", ""))

    # Create the summary dictionary
    summary_dict = {
        "model_name": model.name,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "layers": len(model.layers),
        "summary_text": "\n".join(summary_list)
    }

    return summary_dict