import numpy as np
from src.leak_model import BackgroundLeakSegmenter
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
import lightning as L


class FrameDataset(Dataset):
    def __init__(self, frames_array):
        # Convert numpy frames to tensor format
        frames_rgb = frames_array[:, :, :, ::-1]  # BGR to RGB
        # Use torch.from_numpy with copy to avoid negative stride issues
        self.frames = (
            torch.from_numpy(frames_rgb.copy()).float().permute(0, 3, 1, 2) / 255.0
        )

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return {"frames": self.frames}


def reconstruct(frames: list) -> np.ndarray:
    frames_array = np.array(frames)

    model = BackgroundLeakSegmenter.load_from_checkpoint(
        "lightning_logs/version_5/checkpoints/best-epoch=23-val_loss=0.155.ckpt"
    )
    trainer = L.Trainer(
        accelerator="auto",
        # devices=1,
        logger=False,
    )

    dataset = FrameDataset(frames_array)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        # num_workers=0,  # Set to 1 to avoid shared memory issues
        pin_memory=False,  # Also disable pin_memory to save memory
    )

    predictions = trainer.predict(model, dataloader)
    reconstructed_tensor = predictions[0][0]  # type: ignore

    reconstructed_np = reconstructed_tensor.permute(1, 2, 0).numpy() * 255
    reconstructed_bg = cv2.cvtColor(
        reconstructed_np.astype(np.uint8), cv2.COLOR_RGB2BGR
    )

    return reconstructed_bg
