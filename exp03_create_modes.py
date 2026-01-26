from src.video_dataset import VideoFrameDataset
from src.util import load_triplets

triplets = load_triplets(
    "data/public/backgrounds", "data/public/videos", "data/public/masks"
)

train_dataset = VideoFrameDataset(
    triplets,
    threshold=0.0,
    use_edge=True,
    use_dog=True,
    use_mode=True,
    mode_dir="data/public/modes",
    mode="train",
)
