import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple
import gc


from src.video_dataset import (
    preprocess_frame,
    frame_to_tensor,
    BackgroundLeakDataModule,
    VideoFrameDataset,
)

# from src.util import (load_triplets, load_background, load_mask, metric_CIEDE2000, evaluate)
from src.leak_model import BackgroundLeakSegmenter


def test_model():
    """Main testing function"""

    # Path to the best checkpoint (choose the best one from version_5)
    checkpoint_path = Path(
        "lightning_logs/version_5/checkpoints/best-epoch=22-val_loss=0.155.ckpt"
    )

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for ckpt in Path("lightning_logs/version_5/checkpoints/").glob("*.ckpt"):
            print(f"  {ckpt}")
        return

    # Initialize data module for testing
    data_module = BackgroundLeakDataModule(
        backgrounds_dir="data/public/backgrounds",
        videos_dir="data/public/videos",
        masks_dir="data/public/masks",
        batch_size=1,  # Test one video at a time
        num_workers=4,
        frames_per_video=0,  # Use all frames for testing
    )
    data_module.setup(stage="test")

    # Load model from checkpoint
    print(f"Loading model from: {checkpoint_path}")
    model = BackgroundLeakSegmenter.load_from_checkpoint(str(checkpoint_path))

    # Initialize trainer for testing
    trainer = L.Trainer(
        accelerator="auto", devices=1, logger=False, enable_progress_bar=True
    )

    # Run testing
    print("Starting background reconstruction testing...")
    test_results = trainer.test(model, datamodule=data_module)

    print(f"\n=== Final Test Results ===")
    for result in test_results:
        for key, value in result.items():
            print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    # Clear memory
    gc.collect()
    torch.cuda.empty_cache()

    test_model()
