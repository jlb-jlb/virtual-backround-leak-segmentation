import os

import lightning as L
from torch.utils.data import Dataset
from pathlib import Path



class FrameDataset(Dataset):
    """Dataset for loading the Video Frames. It doesn't load the video frames directly, 
    but keeps track of the frame indices. 
    It loads the true background frames into memory
    loads the mode into memory
    given parameter: 
        path

    Args:
        Dataset (_type_): _description_
        paths: Required path: video, background, mask Optional: mode, edge, dog
        use_bgr: If true uses the original BGR frames
        use_edge: If true uses the Canny edge filter
        use_dog: If true uses the Difference of Gaussian filter
        use_mode: If true uses the mode image

    """

    def __init__(
            self,
            paths: list[dict[str, str]],
            frames_per_video: int = 300,
            *,
            use_bgr: bool = False,
            use_edge: bool = False,
            use_dog: bool = False,
            use_mode: bool = False,
            verbose: bool = False,
    ):
        
        self.paths = paths
        self.frames_per_video = frames_per_video
        self._mode_cache = {}
        



    def _load_mode_img(self, video_path: str|Path):
        key = str(video_path)
        if key in self._mode_cache:
            return self._mode_cache[key]
        # else
        video_name = os.path.splitext(os.path.basename(key))[0]
        mode_path = os.path.join(self.mode_dir, f"{video_name}_mode.png")
        if not os.path.exists(mode_path):
            raise FileNotFoundError(f"Mode image not found at {mode_path}")
        