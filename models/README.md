# Model Files

This directory is used to store trained model files. The training scripts will save models here by default.

## Expected Model Files

After training, you should find files like:

- `damage_classifier_xception.h5` - Xception-based model
- `damage_classifier_inception_v3.h5` - InceptionV3-based model
- `damage_classifier_inception_resnet_v2.h5` - Inception-ResNet-V2-based model

## Model Selection

When evaluating, you can specify which model to use with the `--model_path` argument, for example:

```bash
python src/evaluate.py --model_path models/damage_classifier_xception.h5
```

## Sharing Models

Trained models can be several hundred MB in size. If you need to share them with team members, consider:

1. Using Git LFS if your repository supports it
2. Sharing via cloud storage
3. Providing instructions for team members to train their own models

## Pre-trained Weights

All models are initialized with weights pre-trained on ImageNet. The training scripts will automatically download these weights if they're not already available.