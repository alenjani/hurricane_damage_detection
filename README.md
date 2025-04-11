# Building Damage Classifier

This repository contains code for automatically classifying building images as having "major damage" or "non-major damage" after natural disasters, based on the approach described in the paper "Towards fully automated post-event data collection and analysis: pre-event and post-event information fusion" by Lenjani et al.

## Overview

The system uses deep convolutional neural networks (CNNs) to classify building images and an information fusion approach to combine predictions from multiple images of the same building to make a robust decision.

### Features

- Train damage classification models using multiple CNN architectures (Xception, InceptionV3, InceptionResNetV2)
- Fuse information from multiple building images to make robust predictions
- Evaluate model performance with visualization tools

## Installation

```bash
# Clone the repository
git clone https://github.com/alenjani/hurricane_damage_detection.git
cd hurricane-damage-classifier

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Dataset Structure

Place your source image dataset in the following structure:

```
data/
└── raw/
    ├── major_damage/     # Images of buildings with major damage
    └── non_major_damage/ # Images of buildings with non-major damage
```

## Usage

### 1. Data Preparation

Split your raw dataset into training and validation sets:

```bash
python src/data_processing.py
```

### 2. Training Models

Train a model (Xception by default):

```bash
python src/train.py --model xception --epochs 100 --batch_size 16
```

Available model options: `xception`, `inception_v3`, `inception_resnet_v2`

### 3. Evaluation

Evaluate the model on validation data:

```bash
python src/evaluate.py --model_path models/damage_classifier_xception.h5
```

### 4. Building-Level Evaluation

Evaluate on multiple images of the same building:

```bash
python src/validate_buildings.py --model_path models/damage_classifier_xception.h5 --buildings_dir data/validation_buildings
```

## Citation

If you use this code in your research, please cite the original paper:

```
Lenjani, A., Dyke, S. J., Bilionis, I., Yeum, C. M., Kamiya, K., Choi, J., Liu, X., & Chowdhury, A. G. (2019). 
Towards fully automated post-event data collection and analysis: pre-event and post-event information fusion. 
Engineering Structures.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.