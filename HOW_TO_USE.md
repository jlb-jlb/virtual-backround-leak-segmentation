# How to Use VADER - Video Background Reconstruction

This guide provides step-by-step instructions on how to set up, use, and understand the VADER repository for reconstructing leaking backgrounds from video frames.

## 1. Prerequisites & Installation

### System Requirements
- OS: Linux (recommended)
- Python: >= 3.12
- CUDA (optional but recommended for GPU acceleration)

### Installation Steps

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd wave2_vader
    ```

2.  **Install Dependencies:**
    You can use the provided setup script which installs dependencies via `pip`:
    ```bash
    ./setup_script.sh
    ```
    
    Alternatively, you can manually install the requirements:
    ```bash
    pip install -r ./setup/requirements.txt
    ```

    *Note: The project uses `pyproject.toml` and `poetry.lock` if you prefer using Poetry for dependency management.*

## 2. Using the Module (Inference)

The main entry point for using the trained model to reconstruct backgrounds is `solution.py`.

### Running the Example
You can run the script directly to see an example reconstruction on a test video:

```bash
python solution.py
```

This will:
1.  Load the pre-trained model (from `lightning_logs/version_24/`).
2.  Load a sample video (`data/public/videos/2_i_kitchen_horses_mp.mp4`).
3.  Reconstruct the background.
4.  Save the result as `reconstructed_background_exp4.jpg`.

### Integrating into Your Code
To use the reconstruction logic in your own scripts, import the `reconstruct` function from `solution.py`.

```python
from solution import reconstruct
from src.util import load_video
import cv2

# 1. Load video frames (returns a list of numpy arrays)
frames = load_video("path/to/your/video.mp4")

# 2. Run reconstruction
# This function handles resizing (default 1280x720) and model inference
reconstructed_bg = reconstruct(frames)

# 3. Save or process the result
cv2.imwrite("output_background.jpg", reconstructed_bg)
```

**Function Signature:**
- `reconstruct(frames: list)`: Accepts a list of BGR image frames (numpy arrays) and returns a single BGR image (numpy array) of the reconstructed background.

## 3. Project Structure

- **`src/`**: Contains the source code for the deep learning models and data processing.
    - `leak_model.py`: Defines the `BackgroundLeakSegmenter` PyTorch Lightning module and `UNetSimple` architecture.
    - `video_dataset.py`: Handles data loading, preprocessing (resizing to 1280x720), and filtering (edge detection, DoG).
    - `losses.py`: Custom loss functions (ContinuousRegressionLoss).
    - `util.py`: Helper functions for loading videos and images.
- **`solution.py`**: The main interface for running inference using the best checkpoint.
- **`data/`**: Directory for storing input videos and ground truth backgrounds.
    - `public/videos/`: Input MP4 files.
    - `public/backgrounds/`: Ground truth background images.
- **`lightning_logs/`**: Stores training logs and model checkpoints (`.ckpt` files).
    - The active model uses `version_24`.

## 4. Training (Advanced)

If you wish to retrain the model or explore the training process:

- The model uses **PyTorch Lightning**.
- Training logic is encapsulated in `src/leak_model.py`.
- Experiment notebooks (e.g., `03_image_segmentation.ipynb`, `04_imageSeg.ipynb`) show how the training data was explored and prepared.
- **Training Script**: `exp04_unet.py` is the script used to train the current best model (U-Net architecture with Edge and DoG filters). You can use this as a reference for starting new training experiments.
- To start a new training run, you would typically use the `BackgroundLeakSegmenter` class and a PyTorch Lightning Trainer, similar to how it is used in the notebooks or experiment scripts (`exp*.py`).

## 5. Troubleshooting

- **Memory Issues**: Processing long videos may consume significant RAM. The `reconstruct` function processes frames in batches (internally handled by the model's `reconstruct_from_numpy_frames` method) to mitigate this, but ensure you have enough memory for the loaded video frames.
- **Model Checkpoint**: Ensure that the checkpoint file exists at `lightning_logs/version_24/checkpoints/best-epoch=1-val_loss=0.163.ckpt`. If it is missing, you may need to download the pretrained weights or retrain the model.
- **Input Dimensions**: The model expects inputs resized to `1280x720`. The `reconstruct` function handles this automatically, but be aware if you are feeding manual tensors.
