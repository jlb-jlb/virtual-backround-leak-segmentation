from src.video_dataset import (
    BackgroundLeakDataModule,
    # VideoFrameDataset,
    # preprocess_frame,
    # frame_to_tensor
)

# from src.losses import ContinuousRegressionLoss
# from src.masks import create_continuous_similarity_mask
from src.leak_model import BackgroundLeakSegmenter

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)

MODEL_PATH = "lightning_logs/version_21/checkpoints/best-epoch=1-val_loss=0.153.ckpt"

def train_model(test=False):
    data_module = BackgroundLeakDataModule(
        backgrounds_dir="data/public/backgrounds",
        videos_dir="data/public/videos",
        masks_dir="data/public/masks",
        batch_size=4,  # Small batch size for memory efficiency
        num_workers=8,  # Adjusted for my laptop
        frames_per_video=400,  # Limit frames per video
        threshold=0.0,  # Threshold for similarity mask
        use_edge=True,  # include Canny edge filter
        use_dog=True,  # include DoG filter
    )

    # Model input channels: 3 RGB + edge + DoG = 5
    model = BackgroundLeakSegmenter(
        learning_rate=1e-4,
        loss_type="combined",
        model_architecture="convnet_simple",
        in_channels=8,  # RGB + edge + DoG + mode
        height=720,
        width=1280,
        use_dog=True,  # Ensure DoG filter is used
        use_edge=True,  # Ensure edge filter is used
        use_mode=True
    )

    # Callbacks for memory management and performance
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, verbose=True),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=3,
            mode="min",
            filename="best-{epoch}-{val_loss:.3f}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer with memory optimizations
    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",  # Automatically select GPU/CPU
        devices=1,  # Use single device to avoid memory issues
        precision="16-mixed",  # Mixed precision for memory efficiency
        gradient_clip_val=1.0,  # Prevent exploding gradients
        accumulate_grad_batches=4,  # Simulate larger batch size
        callbacks=callbacks,
        val_check_interval=0.5,  # Check validation twice per epoch
        log_every_n_steps=10,
        enable_progress_bar=True,
        # REMOVED: enable_checkpointing=False,  # This was causing the conflict
        fast_dev_run=False,  # Set to True for debugging
    )

    torch.set_float32_matmul_precision("medium")
    # trainer.fit(model, data_module)

    trainer.test(
        model,
        data_module,
        ckpt_path=MODEL_PATH,
    )


if __name__ == "__main__":
    # Set memory optimization flags
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

    # Enable garbage collection
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    train_model()
