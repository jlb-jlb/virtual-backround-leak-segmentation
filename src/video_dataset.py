import torch  # only torch import needed
import cv2
import numpy as np
import random
import lightning as L
import os
from tqdm import tqdm

from .util import load_triplets
from .masks import create_continuous_similarity_mask


def preprocess_frame(frame, frame_size=(1280, 720)):
    """
    Shared preprocessing function for both training and testing
    Args:
        frame: Input frame from cv2
        frame_size: (width, height) - cv2 convention
    Returns:
        frame_rgb: RGB frame with shape (height, width, 3)
    """
    width, height = frame_size
    frame_resized = cv2.resize(
        frame, (width, height)
    )  # cv2.resize expects (width, height)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return frame_rgb  # Shape: (height, width, 3)


def frame_to_tensor(frame_rgb):
    """
    Convert preprocessed frame to tensor (same as training)
    Args:
        frame_rgb: RGB frame with shape (height, width, 3)
    Returns:
        tensor: Shape (3, height, width) normalized to [0,1]
    """
    return torch.FloatTensor(frame_rgb).permute(2, 0, 1) / 255.0


# Add edge and DoG filter helpers


def edge_filter(frame, low_threshold=50, high_threshold=150):
    # Convert to grayscale (frame is RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def dog_filter(frame, ksize1=5, sigma1=1.0, ksize2=9, sigma2=2.0):
    # Difference of Gaussians filter
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur1 = cv2.GaussianBlur(gray, (ksize1, ksize1), sigma1)
    blur2 = cv2.GaussianBlur(gray, (ksize2, ksize2), sigma2)
    dog = blur1.astype(np.float32) - blur2.astype(np.float32)
    dog_norm = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)  # type: ignore
    return dog_norm.astype(np.uint8)


def mode_image(frame_array: np.ndarray):
    """Input: (Frames, Height, Width, Channels)"""
    # if len(frame_array) == 0:
    #     raise ValueError("Input frame array is empty")
    # Compute mode across frames
    mode_tensor, _ = torch.mode(torch.from_numpy(frame_array), dim=0)
    return mode_tensor



class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        triplets,
        frame_size=(1280, 720),
        frames_per_video=50,
        similarity_method="combined",
        threshold: float = 0.0,
        use_edge: bool = False,
        use_dog: bool = False,
        mode: str = "train",  # train or test
        use_mode: bool = False,  # use mode image as extra channel
        mode_dir: str = None,  # type: ignore
        verbose: bool = False,
    ):
        """
        Args:
            threshold: similarity threshold below which mask values are set to zero.
           use_edge: whether to include Canny edge filter as extra channel.
           use_dog: whether to include Difference of Gaussians filter channel.
        """
        self.triplets = triplets
        self.frame_size = frame_size  # (width, height) convention
        self.frames_per_video = frames_per_video
        self.similarity_method = similarity_method
        self.threshold = threshold
        self.mode = mode  # 'train' or 'test' (Not for mode image computation)
        self.use_edge = use_edge
        self.use_dog = use_dog
        self.use_mode = use_mode
        # Directory to store or load precomputed mode images
        self.mode_dir = mode_dir or os.path.join(os.getcwd(), "mode_images")
        if self.use_mode:
            os.makedirs(self.mode_dir, exist_ok=True)
        # Initialize background cache
        self._bg_cache = {}
        # Initialize mode cache
        self._mode_cache = {}
        # Precompute mode images for all videos to avoid on-the-fly computation

        self.verbose = verbose

        if self.use_mode:
            unique_videos = {vp for vp, _, _ in self.triplets}

            if self.verbose:
                iterable = tqdm(unique_videos, desc="Loading mode images", unit="video")
            else:
                iterable = iter(unique_videos)

            for vp in iterable:
                self._load_mode_image(vp)

        if mode == "test":
            self.frame_indices = []
        else:
            self.frame_indices = []
            self._build_frame_index()

    def _build_frame_index(self):
        """Build index of all available frames across videos"""
        for triplet_idx, (video_path, mask_path, bg_path) in enumerate(self.triplets):
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if total_frames > self.frames_per_video:
                frame_step = total_frames // self.frames_per_video
                selected_frames = list(range(0, total_frames, frame_step))[
                    : self.frames_per_video
                ]
            else:
                selected_frames = list(range(total_frames))

            for frame_idx in selected_frames:
                self.frame_indices.append((triplet_idx, frame_idx))

    def __len__(self):
        if self.mode == "test":
            return len(self.triplets)
        else:
            return len(self.frame_indices)

    def __getitem__(self, idx):
        if self.mode == "test":
            video_path, mask_path, bg_path = self.triplets[idx]
            return {
                "video_path": str(video_path),
                "background_path": str(bg_path),
                "mask_path": str(mask_path),
            }
        else:
            triplet_idx, frame_idx = self.frame_indices[idx]
            video_path, mask_path, bg_path = self.triplets[triplet_idx]

            frame = self._load_single_frame(video_path, frame_idx)
            background = self._load_background(bg_path)
            similarity_mask = create_continuous_similarity_mask(
                frame,
                background,
                method=self.similarity_method,
                threshold=self.threshold,
            )

            # Build frame tensor with optional filter channels
            base_tensor = frame_to_tensor(frame)
            channels = [base_tensor]
            if self.use_edge:
                edges = edge_filter(frame)
                edge_tensor = torch.FloatTensor(edges).unsqueeze(0) / 255.0
                channels.append(edge_tensor)
            if self.use_dog:
                dog = dog_filter(frame)
                dog_tensor = torch.FloatTensor(dog).unsqueeze(0) / 255.0
                channels.append(dog_tensor)
            # Append mode channel if enabled
            if self.use_mode:
                mode_img = self._load_mode_image(video_path)
                mode_tensor = frame_to_tensor(mode_img)
                channels.append(mode_tensor)
            frame_tensor = torch.cat(channels, dim=0)
            return {
                "frame": frame_tensor,
                "similarity_mask": torch.FloatTensor(similarity_mask).unsqueeze(0),
                "background": frame_to_tensor(background),
            }

    def _load_single_frame(self, video_path, frame_idx):
        """Load single frame from video efficiently"""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")

        return preprocess_frame(frame, self.frame_size)

    def _load_background(self, bg_path):
        """Load and cache background images"""
        if str(bg_path) not in self._bg_cache:
            bg = cv2.imread(str(bg_path))
            if bg is None:
                raise ValueError(f"Could not read background image from {bg_path}")
            bg_processed = preprocess_frame(bg, self.frame_size)
            self._bg_cache[str(bg_path)] = bg_processed

        return self._bg_cache[str(bg_path)]

    def _load_mode_image(self, video_path):
        """Load or compute and cache the mode image for a given video.
        ATTENTION: THIS CREATING THE MODE IMAGES SIMULTANEOUSLY WITH TRAINING IS NOT ADVISED Crashy crash crash boom your pc is oom!
        """
        key = str(video_path)
        if key not in self._mode_cache:
            # Mode image file path
            video_name = os.path.splitext(os.path.basename(key))[0]
            mode_path = os.path.join(self.mode_dir, f"{video_name}_mode.png")
            if self.verbose:
                print(f"Loading mode image for {video_name} from {mode_path}")
            # Compute mode if not exists
            if not os.path.exists(mode_path):
                cap = cv2.VideoCapture(str(video_path))
                frames = []
                while True:
                    ret, frm = cap.read()
                    if not ret:
                        break
                    rgb = preprocess_frame(frm, self.frame_size)
                    frames.append(frame_to_tensor(rgb))
                cap.release()
                if frames:
                    stack = torch.stack(frames)  # (F, C, H, W)
                    mode_tensor, _ = torch.mode(stack, dim=0)
                    mode_np = (mode_tensor.permute(1, 2, 0).numpy() * 255).astype(
                        np.uint8
                    )
                else:
                    h, w = self.frame_size[1], self.frame_size[0]
                    mode_np = np.zeros((h, w, 3), dtype=np.uint8)
                # Save mode image in BGR format
                cv2.imwrite(mode_path, cv2.cvtColor(mode_np, cv2.COLOR_RGB2BGR))
                # Free intermediate tensors to release memory
                del frames
                if "stack" in locals():
                    del stack
                if "mode_tensor" in locals():
                    del mode_tensor
                if "mode_np" in locals():
                    del mode_np
                import gc

                gc.collect()
            # Load and preprocess saved mode image

            # return # just to create the images!

            loaded = cv2.imread(mode_path)
            if loaded is None:
                raise ValueError(f"Could not read mode image from {mode_path}")
            proc = preprocess_frame(loaded, self.frame_size)
            self._mode_cache[key] = proc
        return self._mode_cache[key]


# %%
class BackgroundLeakDataModule(L.LightningDataModule):
    def __init__(
        self,
        backgrounds_dir: str,
        videos_dir: str,
        masks_dir: str,
        batch_size: int = 8,
        num_workers: int = 16,
        frames_per_video: int = 50,
        frame_size: tuple = (1280, 720),  # (width, height)
        val_split: float = 0.2,
        threshold: float = 0.0,
        use_edge: bool = False,
        use_dog: bool = False,
        use_mode: bool = False,
        mode_dir: str = None,  # type: ignore
    ):
        super().__init__()
        self.backgrounds_dir = backgrounds_dir
        self.videos_dir = videos_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frames_per_video = frames_per_video
        self.frame_size = frame_size  # (width, height)
        self.val_split = val_split
        self.threshold = threshold
        self.use_edge = use_edge
        self.use_dog = use_dog
        self.use_mode = use_mode
        self.mode_dir = mode_dir

    def setup(self, stage: str = None):  # type: ignore
        triplets = load_triplets(self.backgrounds_dir, self.videos_dir, self.masks_dir)
        random.shuffle(triplets)
        split_idx = int(len(triplets) * (1 - self.val_split))

        if stage == "fit" or stage is None:
            self.train_dataset = VideoFrameDataset(
                triplets[:split_idx],
                frame_size=self.frame_size,
                frames_per_video=self.frames_per_video,
                similarity_method=(
                    self.test_dataset.similarity_method
                    if hasattr(self, "test_dataset")
                    else "combined"
                ),
                threshold=self.threshold,
                use_edge=self.use_edge,
                use_dog=self.use_dog,
                use_mode=self.use_mode,
                mode_dir=self.mode_dir,
                mode="train",
            )
            self.val_dataset = VideoFrameDataset(
                triplets[split_idx:],
                frame_size=self.frame_size,
                frames_per_video=self.frames_per_video // 2,
                similarity_method=(
                    self.test_dataset.similarity_method
                    if hasattr(self, "test_dataset")
                    else "combined"
                ),
                threshold=self.threshold,
                use_edge=self.use_edge,
                use_dog=self.use_dog,
                use_mode=self.use_mode,
                mode_dir=self.mode_dir,
                mode="train",
            )

        if stage == "test" or stage is None:
            self.test_dataset = VideoFrameDataset(
                triplets,  # Use all triplets for testing
                frame_size=self.frame_size,
                threshold=self.threshold,
                use_edge=self.use_edge,
                use_dog=self.use_dog,
                use_mode=self.use_mode,
                mode_dir=self.mode_dir,
                mode="test",
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers // 2,
            pin_memory=True,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset, batch_size=1, shuffle=False, num_workers=1
        )
