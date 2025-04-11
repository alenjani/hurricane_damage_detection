# Dataset for Building Damage Classification

## Directory Structure

This directory should contain the following structure:

```
data/
├── raw/                        # Original source images
│   ├── major_damage/           # Images showing major building damage
│   └── non_major_damage/       # Images showing non-major or no building damage
│
└── processed/                  # Created by data_processing.py
    ├── train/
    │   ├── major_damage/
    │   └── non_major_damage/
    └── val/
        ├── major_damage/
        └── non_major_damage/
```

## Dataset Preparation

1. Place your source images in the `raw/major_damage` and `raw/non_major_damage` directories.

2. Run the data processing script to split the data into training and validation sets:
   ```
   python src/data_processing.py
   ```

3. This will create the processed directory structure with train/validation splits.

## Image Naming Convention

Each image should ideally be named with a building identifier to allow for grouping multiple images of the same building:

```
building_id__image_id.jpg
```

This naming convention helps the information fusion algorithm combine predictions from multiple images of the same building.

## Sharing Large Datasets

Due to size limitations of Git repositories, it's recommended to share the image dataset separately using:

1. Cloud storage (Google Drive, Dropbox)
2. Academic data repositories (Zenodo, OSF.io)
3. DVC (Data Version Control)

For collaboration, provide team members with the dataset download link and instructions to place the images in the proper folders before running the processing script.