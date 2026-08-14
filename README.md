# Brain MRI Tumor Segmentation Using Computer Vision and Deep Learning

A computer vision and biomedical imaging project for automatic tumor-region segmentation from brain MRI slices using OpenCV preprocessing and a lightweight PyTorch U-Net.

## Overview

This project develops an end-to-end biomedical image segmentation pipeline that takes a brain MRI slice as input and predicts the tumor region at pixel level.

The system combines classical image processing with deep learning:

MRI Image
↓
OpenCV Preprocessing
↓
U-Net Segmentation
↓
Tumor Mask
↓
Quantitative Analysis

## Key Features

- Brain MRI preprocessing using OpenCV
- Median filtering for noise reduction
- CLAHE for local contrast enhancement
- Patient-level train/validation/test splitting
- Lightweight U-Net segmentation model
- GPU-accelerated training with PyTorch
- Dice and IoU evaluation
- Precision and recall analysis
- Threshold optimization using validation data
- Streamlit-based interactive inference application
- Tumor-area estimation from predicted segmentation masks

## Dataset

The project uses the LGG Brain MRI Segmentation dataset containing brain MRI slices and corresponding segmentation masks.

The dataset contains:

- 110 patients
- 3,929 MRI slices
- 3,929 corresponding masks
- 1,373 tumor-positive slices
- 2,556 tumor-negative slices
- Image resolution: 256 × 256

The dataset is not included in this repository.

## Dataset Samples

![Dataset Samples](results/01_dataset_samples.png)

## Preprocessing

![Preprocessing](results/02_preprocessing.png)

## Test Predictions

![Test Predictions](results/03_test_predictions.png)

## Data Splitting

To prevent patient-level data leakage, the dataset was divided by patient rather than by individual MRI slice.

| Split | Patients | Slices |
|---|---:|---:|
| Training | 77 | 2,719 |
| Validation | 16 | 621 |
| Test | 17 | 589 |

No patient appears in more than one split.

## Image Preprocessing

The preprocessing pipeline uses OpenCV:

1. Median filtering
2. CLAHE contrast enhancement
3. Pixel normalization to [0, 1]

## Model

A lightweight U-Net architecture was implemented using PyTorch.

### Architecture

```text
Input MRI
    │
    ▼
Encoder
32 → 64 → 128 → 256
    │
    ▼
Bottleneck
512 channels
    │
    ▼
Decoder
256 → 128 → 64 → 32
    │
    ▼
Tumor Segmentation Mask

The model contains approximately 7.8 million trainable parameters.

Training

The model was trained using:

PyTorch
Adam optimizer
Learning rate: 1e-4
Batch size: 8
BCE + Dice loss
Early stopping
Best-model checkpointing
NVIDIA RTX 4060 GPU
Evaluation

The best model was selected using validation Dice score and evaluated on previously unseen test patients.

Test Results

Using a 0.3 segmentation threshold:

Metric	Score
Overall Dice	0.7529
Overall IoU	0.6037
Tumor-positive Dice	0.5690
Tumor-positive IoU	0.4939
Precision	0.8691
Recall	0.5692
Experiment: Class Imbalance

The training data contained substantially more background pixels than tumor pixels.

A weighted BCE + Dice experiment was performed using a training-set-derived positive class weight.

The weighted-loss experiment achieved a lower best validation Dice of 0.5647 compared with 0.7607 for the baseline BCE + Dice approach.

Therefore, the baseline BCE + Dice loss was retained.

Results
Preprocessing

Test Predictions

Streamlit Application

The project includes an interactive Streamlit application.

Users can:

Upload a brain MRI slice
View the original image
Apply preprocessing
Generate a tumor segmentation
View the predicted tumor mask
View the segmentation overlay
Estimate the percentage of the slice classified as tumor

Brain-MRI-Tumor-Segmentation/
│
├── app.py
├── README.md
├── requirements.txt
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── split.py
│   ├── dataloader.py
│   ├── model.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── final_evaluation.py
│   ├── visualize_predictions.py
│   └── find_threshold.py
│
├── models/
├── results/
└── Datasets/

Limitations
The model operates on 2D MRI slices rather than complete 3D MRI volumes.
The current model is trained for tumor-region segmentation rather than tumor subtype classification.
Test performance indicates that some smaller tumor regions are missed.
The dataset is limited to the selected MRI segmentation dataset and may not represent variation across scanners, institutions, or acquisition protocols.
Future Improvements
Extend the pipeline to 3D volumetric MRI segmentation
Experiment with attention U-Net or transformer-based segmentation architectures
Add data augmentation
Improve small-lesion segmentation
Evaluate cross-dataset generalization
Add uncertainty estimation
Integrate additional biomedical imaging modalities
Technologies
Python
OpenCV
PyTorch
NumPy
Pandas
Matplotlib
Scikit-learn
Streamlit
CUDA

## Dataset

This project uses the LGG Brain MRI Segmentation dataset containing
brain MRI slices and corresponding segmentation masks.

Dataset source:

https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation

The dataset is not included in this repository. Users should obtain
the dataset directly from the original source and comply with its
applicable terms of use.

The underlying TCGA-LGG imaging data is associated with The Cancer
Imaging Archive (TCIA).

Disclaimer

This project is intended for research and educational purposes and is not a clinical diagnostic system.