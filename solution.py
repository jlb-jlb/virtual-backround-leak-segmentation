import numpy as np
from src.leak_model import BackgroundLeakSegmenter
import cv2


# Load model including filter flags
model = BackgroundLeakSegmenter.load_from_checkpoint(
    "lightning_logs/version_24/checkpoints/best-epoch=1-val_loss=0.163.ckpt",
    use_edge=True,
    use_dog=True,
    # use_mode=True,
    # mode_dir="data_dummy/public/modes",
)


def reconstruct(frames: list) -> np.ndarray:
    """
    Build background by running reconstruct_from_numpy_frames on raw BGR frames.
    """
    frames = np.array(frames) # type: ignore
    # print(len(frames))
    # reconstruct_from_numpy_frames handles resizing and filter channels
    reconstructed_bg = model.reconstruct_from_numpy_frames(frames, verbose=False)

    del frames  # Free memory
    
    return reconstructed_bg


if __name__ == "__main__":
    from src.util import load_video

    # Example usage
    frames = load_video("data/public/videos/2_i_kitchen_horses_mp.mp4")  # type: ignore
    # use longest video for testing memory
    reconstructed_bg = reconstruct(frames)
    cv2.imwrite("reconstructed_background_exp4.jpg", reconstructed_bg)
