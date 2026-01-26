import numpy as np
from src.leak_model import BackgroundLeakSegmenter
import cv2


# Load model including filter flags
model = BackgroundLeakSegmenter.load_from_checkpoint(
    "lightning_logs/version_12/checkpoints/best-epoch=2-val_loss=0.174-v1.ckpt",
    use_edge=True,
    use_dog=True,
)


def reconstruct(frames: list) -> np.ndarray:
    """
    Build background by running reconstruct_from_numpy_frames on raw BGR frames.
    """
    frames_array = np.array(frames)
    # reconstruct_from_numpy_frames handles resizing and filter channels
    reconstructed_bg = model.reconstruct_from_numpy_frames(frames_array, verbose=False)
    return reconstructed_bg


if __name__ == "__main__":
    from src.util import load_video

    # Example usage
    frames = load_video("data/public/videos/2_i_kitchen_bridge_mp.mp4")  # type: ignore
    reconstructed_bg = reconstruct(frames)
    cv2.imwrite("reconstructed_background_exp2.jpg", reconstructed_bg)
