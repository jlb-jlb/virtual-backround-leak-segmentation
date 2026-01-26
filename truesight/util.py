import numpy as np
import torch
import cv2
import colour
from pathlib import Path
from typing import List, Tuple

from skimage.color import deltaE_ciede2000

__all__ = [
    "load_video",
    "load_background",
    "load_mask",
    "metric_CIEDE2000",
    "evaluate",
    "load_triplets",
    "show_image",
]

def load_video(video_fpath: Path) -> list:
    vc = cv2.VideoCapture(str(video_fpath))
    counter = 0
    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False

    frames = []
    while rval:
        frames.append(frame)
        counter += 1
        rval, frame = vc.read()
    vc.release()
    return frames


def load_background(bg_fpath: Path|str) -> np.ndarray:
    background = cv2.imread(str(bg_fpath))
    return background


def load_mask(mask_fpath: Path) -> np.ndarray:
    """Mask is range 0 - 255"""
    mask = cv2.imread(str(mask_fpath), cv2.IMREAD_GRAYSCALE)
    return mask


def metric_CIEDE2000(
    reconstruction: np.ndarray, background: np.ndarray, gt_mask: np.ndarray
) -> np.ndarray:
    gt_mask = gt_mask > 0
    reconstruction = cv2.cvtColor(
        reconstruction.astype(np.float32) / 255, cv2.COLOR_RGB2Lab
    )
    background = cv2.cvtColor(background.astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
    delta_e = colour.delta_E(reconstruction, background, method="CIE 2000")
    delta_e[~gt_mask] = 254.0

    return delta_e


def evaluate(deltae: np.ndarray, mask: np.ndarray) -> float:
    deltae = deltae.flatten()
    assert np.max(mask) == 255 or np.max(mask) == 0
    mask = mask.flatten() / 255.0
    assert min(mask) == 0

    reconstructed = sum(mask[deltae < 4])
    leaking = sum(mask)
    reconstruction_score = reconstructed / leaking

    return reconstruction_score


def show_image(image: np.ndarray | torch.Tensor, title: str = "", cmap=None) -> None:
    """Display image from numpy array or torch tensor. in BGR format.
    uses matplotlib for display.

    Args:
        image (np.ndarray | torch.Tensor): image, can be a numpy array or a torch tensor.
            If a torch tensor, it should be in the format (H, W, C) and in BGR format.
    """
    import matplotlib.pyplot as plt
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    # if 3D
    plt.figure(figsize=(10, 10))
    if image.ndim == 3 and image.shape[2] == 3:
        plt.imshow(image[:,:,::-1]) # Convert BGR to RGB for display
        plt.title(title)
        plt.axis('off')  # Hide axes
    elif image.ndim == 2:  # Grayscale image
        cmap_ = cmap if cmap else 'gray'
        plt.imshow(image, cmap=cmap_)
        plt.title(title)
        plt.axis('off')
    else:
        raise ValueError("Image must be 2D or 3D (H, W, C) format.")
    plt.show()


def _show_image(image: np.ndarray | torch.Tensor, title: str = "", cmap=None) -> None:
    """Display a single image from numpy array or torch tensor in BGR format.
    Uses matplotlib for display.

    Args:
        image (np.ndarray | torch.Tensor): Image, can be a numpy array or a torch tensor.
            If a torch tensor, it should be in the format (H, W, C) and in BGR format.
        title (str, optional): Title for the image. Defaults to "".
        cmap (str, optional): Colormap for grayscale images. Defaults to None.
    """
    import matplotlib.pyplot as plt
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.ndim == 3 and image.shape[2] == 3:
        plt.imshow(image[:, :, ::-1])
        # Convert BGR to RGB for display
        plt.title(title)
        plt.axis('off')
    elif image.ndim == 2:  # Grayscale image
        cmap_ = cmap if cmap else 'gray'
        plt.imshow(image, cmap=cmap_)
        plt.title(title)
        plt.axis('off')
    else:
        raise ValueError("Image must be 2D or 3D (H, W, C) format.")

def show_images(
    images: List[np.ndarray | torch.Tensor],
    titles: List[str] = None, # type: ignore
    cmap=None,
    cols: int = 3,
) -> None:
    """Display multiple images in a grid.

    Args:
        images (List[np.ndarray | torch.Tensor]): List of images to display.
        titles (List[str], optional): List of titles for each image. Defaults to None.
        cmap (str, optional): Colormap for grayscale images. Defaults to None.
        cols (int, optional): Number of columns in the grid. Defaults to 3.
    """
    import matplotlib.pyplot as plt

    if titles is None:
        titles = [""] * len(images)

    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(15, rows * 5))

    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        _show_image(image, title=titles[i], cmap=cmap)

    plt.tight_layout()
    plt.show()






def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 image to float32 CIE L*a*b*:
      L ∈ [0,100], a,b ∈ [-128,127]
    """
    lab_8u = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    lab = lab_8u.astype(np.float32)
    lab[..., 0] = lab[..., 0] * (100.0 / 255.0)
    lab[..., 1:] = lab[..., 1:] - 128.0
    return lab

def delta_e_image(
    frame_bgr: np.ndarray,
    background_bgr: np.ndarray
) -> np.ndarray:
    """
    Compute the per-pixel CIEDE2000 ΔE between two BGR images.
    Returns an HxW float64 array of ΔE values.
    """
    lab1 = bgr_to_lab(frame_bgr)
    lab2 = bgr_to_lab(background_bgr)
    # skimage expects shape (H,W,3)
    return deltaE_ciede2000(lab1, lab2)



# My own functions
def load_triplets(
    backgrounds_dir: str | Path,
    videos_dir: str | Path,
    masks_dir: str | Path,
) -> List[Tuple[Path, Path, Path]]:
    """
    Collect (video, mask, background) filepath triplets that belong together.

    The expected filename patterns are

    * videos:  <person>_<gt_bg>_<virt_bg>_mp.mp4
               e.g. 2_i_kitchen_bridge_mp.mp4
    * masks:   <person>_<gt_bg>_mp.png
               e.g. 2_i_kitchen_mp.png
    * backgrounds: <gt_bg>.png
               e.g. kitchen.png

    Parameters
    ----------
    backgrounds_dir, videos_dir, masks_dir
        Directory paths (str or pathlib.Path).

    Returns
    -------
    list[tuple[pathlib.Path, pathlib.Path, pathlib.Path]]
        Matched (video, mask, background) filepaths.

    Notes
    -----
    * Only videos for which **both** the corresponding mask **and**
      background image exist are returned.
    * You can freely switch to using `os.path` instead of `pathlib.Path`
      if you prefer.
    """
    backgrounds_dir = Path(backgrounds_dir)
    videos_dir = Path(videos_dir)
    masks_dir = Path(masks_dir)

    # Build quick look-up tables for masks and backgrounds
    masks = {p.name: p for p in masks_dir.glob("*.png")}
    bgs = {p.stem: p for p in backgrounds_dir.glob("*.png")}

    triplets: List[Tuple[Path, Path, Path]] = []

    for video_path in videos_dir.glob("*.mp4"):
        stem_parts = video_path.stem.split("_")
        if len(stem_parts) < 5 or stem_parts[-1] != "mp":
            # Skip unexpected file names
            continue

        # Extract pieces
        person = "_".join(stem_parts[:2])  # e.g. "2_i"
        gt_bg = stem_parts[2]  # e.g. "kitchen"
        # virtual_bg = stem_parts[3]  # not needed here

        # Expected companion names
        mask_name = f"{person}_{gt_bg}_mp.png"
        bg_path = bgs.get(gt_bg)

        # Check presence
        mask_path = masks.get(mask_name)
        if mask_path is None or bg_path is None:
            # Incomplete set → ignore or optionally warn
            continue

        triplets.append((video_path, mask_path, bg_path))

    return triplets
