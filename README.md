# VADER: Video Background Reconstruction

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)
![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)

## Overview

**VADER** (Video Artifact Detection & Eradication/Reconstruction) is a computer vision project designed to reconstruct the static background of a video scene where a virtual background has been applied but the real background occasionally "leaks" through.

This repository contains the implementation of a deep learning solution that utilizes a U-Net architecture to identify and reconstruct these leaking background pixels from video frames, effectively recovering the original background behind a moving subject.

## Key Features

*   **Deep Learning Segmentation**: Uses a custom U-Net architecture to segment leaking background pixels from artifacts.
*   **Advanced Feature Engineering**: Enhances input data with Canny Edge detection and Difference of Gaussians (DoG) filters to improve segmentation boundaries.
*   **Custom Loss Function**: Implements a continuous regression loss combining MSE, Gradient Loss, and SSIM (Structural Similarity Index) for fine-grained training stability.
*   **PyTorch Lightning**: Built on PyTorch Lightning for scalable and organized training pipelines.

## Approach & Methodology

The solution evolved from statistical color analysis to a deep learning segmentation approach. This section details the theoretical and practical steps taken to solve the background reconstruction problem.

### 1. Problem Analysis & Hypothesis
The core challenge is identifying "leaking" background pixels in video frames where a virtual background is applied.
*   **Virtual Background**: Creating a mode in the pixel's color distribution over time.
*   **Subject (Person)**: Occluding the background, creating separate distribution peaks.
*   **Leakage**: Brief, often blurred appearances of the real background.

Initial attempts (Solution A) focused on **statistical clustering** of pixel time series (finding color distribution modes). However, the noise from the moving subject made reduced the effectiveness of this approach.

### 2. Deep Learning Segmentation (Solution B)
The approach shifted to **framewise segmentation** using Convolutional Neural Networks (CNNs). Since exact ground truth labels were unavailable (only flawed masks or the original background images were accessible), we generated **continuous similarity masks** to serve as training targets.

#### Data Preparation
*   **Continuous Similarity Mask**: Instead of a binary mask, which loses information due to the soft edges of background leaks, we created a continuous similarity map.
    *   This map compares the video frame to the ground truth background using perceptual distance metrics (LAB color space).
    *   It serves as a "soft" label for the regression-based training.
*   **Feature Engineering**:
    *   **Edge Detection (Canny)**: Highlights high-frequency structural changes, aiding in boundary detection.
    *   **Difference of Gaussians (DoG)**: approximates Laplacian of Gaussian to detect blobs and edges at different scales.

#### Model Architecture
*   **U-Net**: A simplified U-Net architecture was chosen for its ability to capture both local context and global structure.
    *   **Encoder**: Captures features at multiple scales via 3x3 convolutions and max pooling.
    *   **Decoder**: Upsamples feature maps to reconstruct the segmentation mask.
    *   **Input Channels**: 5 Channels (3 RGB + 1 Edge + 1 DoG).

#### Loss Function
A custom **Continuous Regression Loss** was developed to handle the soft labels:
*   $\alpha \cdot \text{MSE}$ (Mean Squared Error): Basic pixel-wise accuracy.
*   $\beta \cdot \text{Gradient Loss}$: Enforces sharpness at the boundaries.
*   $\gamma \cdot \text{SSIM}$ (Structural Similarity Index): Ensures the structural integrity of the predicted regions.

### 3. Iterative Improvements
*   Experiment 1-3 explored simple CNNs and different filter combinations.
*   Experiment 4 (Current Best): Integrated the U-Net architecture with Edge and DoG filters, achieving significant improvements in test scores.

## Installation

### Prerequisites
*   OS: Linux (Recommended)
*   Python: >= 3.12
*   CUDA (optional, for GPU acceleration)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jlb-jlb/virtual-backround-leak-segmentation.git
    cd virtual-backround-leak-segmentation
    ```

2.  **Install Dependencies:**
    You can use the provided setup script to install all required packages:
    ```bash
    ./setup_script.sh
    ```
    
    Alternatively, install manually via pip:
    ```bash
    pip install -r setup/requirements.txt
    ```
    
    *Note: The project uses `pyproject.toml` or `requirements.txt` for dependency management.*

## Usage

### Inference (Reconstruction)

The main entry point for using the pre-trained model to reconstruct backgrounds is `solution.py`.

1.  **Prepare the Model**: Ensure the trained model checkpoint is located at:
    `lightning_logs/version_24/checkpoints/best-epoch=1-val_loss=0.163.ckpt`

2.  **Run the Script**:
    ```bash
    python solution.py
    ```
    This will load a sample video, run the reconstruction model, and save the output as `reconstructed_background_exp4.jpg`.

### Python API integration

To use the reconstruction logic in your own application:

```python
from solution import reconstruct
from src.util import load_video
import cv2

# 1. Load video frames (returns a list of numpy arrays)
frames = load_video("path/to/your/video.mp4")

# 2. Run reconstruction
# The function handles resizing (1280x720) and inference automatically
reconstructed_bg = reconstruct(frames)

# 3. Save or process the result
cv2.imwrite("output_background.jpg", reconstructed_bg)
```

### Training

To retrain the model or experiment with different architectures:

1.  **Data Setup**: Ensure your training data (backgrounds, videos, masks) is in the `data/` directory as structured in `data_dir.txt`.
2.  **Run Training**:
    The main training script for the best-performing model (Experiment 4) is `exp04_unet.py`.
    ```bash
    python exp04_unet.py
    ```
    This script utilizes the `BackgroundLeakDataModule` and `BackgroundLeakSegmenter` to train the U-Net model.

## Project Structure

```
virtual-backround-leak-segmentation/
├── data/                   # Dataset directory
├── lightning_logs/         # Training logs and model checkpoints
├── src/                    # Core source code
│   ├── leak_model.py       # PyTorch Lightning Module & U-Net implementations
│   ├── video_dataset.py    # Dataset loading, preprocessing, and filtering (Edge/DoG)
│   ├── losses.py           # Custom ContinuousRegressionLoss
│   ├── masks.py            # Similarity mask generation logic
│   └── util.py             # Utility functions (loading, metrics)
├── solution.py             # Inference entry point
├── exp04_unet.py           # Training script for the current best model
└── setup/                  # Installation requirements
```

## Model Performance

The solution was iteratively improved through several experiments. The final selected model uses a **U-Net** architecture with **Edge and DoG filters** as additional input channels.

*   **Metric**: Continuous Regression Loss (Combination of MSE, Gradient, SSIM)
*   **Best Validation Loss**: `0.163`
*   **Test Score**: `0.068` on the VADER challenge test set.

## Authors

*   **Joscha Bisping**

---
*This repository was created as part of the RAID Wave 2 VADER challenge.*