import cv2
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path


def edge_feature(frame, low_threshold=50, high_threshold=150):
    """INPUT BGR Format"""
    # Convert to grayscale (frame is RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges

def dog_feature(frame, ksize1=5, sigma1=1.0, ksize2=9, sigma2=2.0):
    """INPUT BGR Format"""
    # Difference of Gaussians filter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur1 = cv2.GaussianBlur(gray, (ksize1, ksize1), sigma1)
    blur2 = cv2.GaussianBlur(gray, (ksize2, ksize2), sigma2)
    dog = blur1.astype(np.float32) - blur2.astype(np.float32)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    return dog_norm.astype(np.uint8)




def mode_feature(frames: np.ndarray) -> torch.Tensor:
    """Calculate the mode of the frames. from_numpy doesn't allocate new memory!

    Args:
        frames (np.ndarray): Input frames of shape (N, H, W, C) where N is the number of frames,
            H is height, W is width, and C is the number of channels (e.g., 3 for RGB).

    Returns:
        torch.Tensor: Mode of the frames across the first dimension (N), resulting in a tensor of shape (H, W, C).
    """
    mode_tensor, _ = torch.mode(torch.from_numpy(frames), dim=0) 
    return mode_tensor


def load_mode_feature(mode_dir: str|Path) -> np.ndarray:
    mode_img = cv2.imread(str(mode_dir), cv2.IMREAD_UNCHANGED)
    if mode_img is None:
        raise ValueError(f"Mode image not found at {mode_dir}")
    return mode_img

