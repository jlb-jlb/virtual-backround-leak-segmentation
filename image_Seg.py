# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import random
import gc
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)


from util import load_video, load_background, load_mask, metric_CIEDE2000, evaluate, load_triplets  # type: ignore


# %%
def preprocess_frame(frame, frame_size=(720, 1280)):
    """
    Shared preprocessing function for both training and testing
    """
    frame_resized = cv2.resize(frame, frame_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    return frame_rgb


def frame_to_tensor(frame_rgb):
    """
    Convert preprocessed frame to tensor (same as training)
    """
    return torch.FloatTensor(frame_rgb).permute(2, 0, 1) / 255.0


class VideoFrameDataset(Dataset):
    """
    Memory-efficient dataset that loads video frames on-demand[4][8]
    """

    def __init__(
        self,
        triplets,
        frame_size=(720, 1280),
        frames_per_video=50,
        similarity_method="combined",
        mode="train",
    ):  # Add mode parameter
        self.triplets = triplets
        self.frame_size = frame_size
        self.frames_per_video = frames_per_video
        self.similarity_method = similarity_method
        self.mode = mode

        if mode == "test":
            # For testing, we don't build frame indices since we'll process complete videos
            self.frame_indices = []
        else:
            # For training/validation, build frame indices as before
            self.frame_indices = []
            self._build_frame_index()

    def _build_frame_index(self):
        """Build index of all available frames across videos"""
        for triplet_idx, (video_path, mask_path, bg_path) in enumerate(self.triplets):
            # Get video info without loading all frames[3]
            cap = cv2.VideoCapture(str(video_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Sample frame indices evenly across video[8]
            if total_frames > self.frames_per_video:
                frame_step = total_frames // self.frames_per_video
                selected_frames = list(range(0, total_frames, frame_step))[
                    : self.frames_per_video
                ]
            else:
                selected_frames = list(range(total_frames))

            # Add to global frame index
            for frame_idx in selected_frames:
                self.frame_indices.append((triplet_idx, frame_idx))  # type: ignore

    def __len__(self):
        if self.mode == "test":
            return len(self.triplets)  # One sample per video triplet
        else:
            return len(self.frame_indices)  # type: list[Tuple[int, int]]

    def __getitem__(self, idx):
        if self.mode == "test":
            # Return video paths for complete reconstruction
            video_path, mask_path, bg_path = self.triplets[idx]
            return {
                "video_path": str(video_path),
                "background_path": str(bg_path),
                "mask_path": str(mask_path),
            }
        else:
            # Normal frame-by-frame loading for training
            triplet_idx, frame_idx = self.frame_indices[idx]
            video_path, mask_path, bg_path = self.triplets[triplet_idx]

            frame = self._load_single_frame(video_path, frame_idx)
            background = self._load_background(bg_path)
            similarity_mask = create_continuous_similarity_mask(
                frame, background, method=self.similarity_method
            )

            return {
                "frame": frame_to_tensor(frame),  # Consistent tensor conversion
                "similarity_mask": torch.FloatTensor(similarity_mask).unsqueeze(0),
                "background": frame_to_tensor(
                    background
                ),  # Consistent tensor conversion
            }

    def _load_single_frame(self, video_path: Path, frame_idx: int) -> np.ndarray:
        """Load single frame from video efficiently"""
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {video_path}")

        return preprocess_frame(frame, self.frame_size)

    def _load_background(self, bg_path: Path) -> np.ndarray:
        """Load and cache background images"""
        if not hasattr(self, "_bg_cache"):
            self._bg_cache = {}

        if str(bg_path) not in self._bg_cache:
            bg = cv2.imread(str(bg_path))
            if bg is None:
                raise ValueError(f"Could not read background image from {bg_path}")
            bg_processed = preprocess_frame(bg, self.frame_size)
            self._bg_cache[str(bg_path)] = bg_processed

        return self._bg_cache[str(bg_path)]


# %%
class BackgroundLeakDataModule(L.LightningDataModule):
    """Lightning Data Module with efficient video loading[1][2]"""

    def __init__(
        self,
        backgrounds_dir: str,
        videos_dir: str,
        masks_dir: str,
        batch_size: int = 4,  # Smaller batch for memory efficiency
        num_workers: int = 8,  # adjusted for my laptop
        frames_per_video: int = 50,
        val_split: float = 0.2,
    ):
        super().__init__()
        self.backgrounds_dir = backgrounds_dir
        self.videos_dir = videos_dir
        self.masks_dir = masks_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frames_per_video = frames_per_video
        self.val_split = val_split

    def prepare_data(self):
        """Called only once to prepare data[1]"""
        # Verify all files exist
        triplets = load_triplets(self.backgrounds_dir, self.videos_dir, self.masks_dir)
        print(f"Found {len(triplets)} video triplets")

    def setup(self, stage: str = None):  # type: ignore
        triplets = load_triplets(self.backgrounds_dir, self.videos_dir, self.masks_dir)
        random.shuffle(triplets)
        split_idx = int(len(triplets) * (1 - self.val_split))

        if stage == "fit" or stage is None:
            self.train_dataset = VideoFrameDataset(
                triplets[:split_idx],
                frames_per_video=self.frames_per_video,
                mode="train",
            )
            self.val_dataset = VideoFrameDataset(
                triplets[split_idx:],
                frames_per_video=self.frames_per_video // 2,
                mode="train",  # Validation still uses frame-by-frame
            )

        if stage == "test" or stage is None:
            self.test_dataset = VideoFrameDataset(
                triplets[split_idx:],
                mode="test",  # Test mode for complete video reconstruction
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,  # Faster GPU transfer[2]
            persistent_workers=True,  # Keep workers alive[1]
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers // 2,  # Fewer workers for validation
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,  # Process one video at a time
            shuffle=False,
            num_workers=1,
        )


# %%


class ContinuousRegressionLoss(nn.Module):
    """
    Custom loss function for continuous similarity targets.
    Supports multiple loss types optimized for soft label regression[1][4].
    """

    def __init__(self, loss_type="combined", alpha=2.0, beta=1.0, gamma=0.5):
        super().__init__()
        self.loss_type = loss_type
        self.alpha = alpha  # Weight for primary loss
        self.beta = beta  # Weight for gradient loss
        self.gamma = gamma  # Weight for structural loss

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model output [B, 1, H, W] in range [0, 1]
            targets: Continuous similarity masks [B, 1, H, W] in range [0, 1]
        """
        if self.loss_type == "mse":
            return F.mse_loss(predictions, targets)

        elif self.loss_type == "mae":
            # Mean Absolute Error - robust to outliers[5]
            return F.l1_loss(predictions, targets)

        elif self.loss_type == "smooth_l1":
            # Smooth L1 - less sensitive to outliers than MSE[1]
            return F.smooth_l1_loss(predictions, targets)

        elif self.loss_type == "focal_mse":
            # Focal-style loss for continuous targets
            mse = (predictions - targets) ** 2
            # Focus more on difficult pixels (high error)
            focal_weight = (mse + 1e-8) ** (self.alpha / 2)
            return torch.mean(focal_weight * mse)

        elif self.loss_type == "jaccard_soft":
            # Soft Jaccard loss for continuous targets[3]
            intersection = torch.sum(predictions * targets, dim=(2, 3))
            union = torch.sum(predictions + targets - predictions * targets, dim=(2, 3))
            jaccard = intersection / (union + 1e-8)
            return 1 - torch.mean(jaccard)

        elif self.loss_type == "combined":
            # Combine multiple loss components for robust training
            return self._combined_loss(predictions, targets)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def _combined_loss(self, predictions, targets):
        """
        Combined loss with multiple components for robust training[4]
        """
        # Primary regression loss (MSE)
        mse_loss = F.mse_loss(predictions, targets)

        # Gradient-based boundary loss
        grad_loss = self._gradient_loss(predictions, targets)

        # Structural similarity component
        ssim_loss = self._ssim_loss(predictions, targets)

        # Weighted combination
        total_loss = (
            self.alpha * mse_loss + self.beta * grad_loss + self.gamma * ssim_loss
        )

        return total_loss

    def _gradient_loss(self, predictions, targets):
        """
        Gradient-based loss to preserve edge information[4]
        """
        # Sobel operators for gradient computation
        sobel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(
            1, 1, 3, 3
        )
        sobel_y = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(
            1, 1, 3, 3
        )

        if predictions.is_cuda:
            sobel_x = sobel_x.cuda()
            sobel_y = sobel_y.cuda()

        # Compute gradients
        grad_pred_x = F.conv2d(predictions, sobel_x, padding=1)
        grad_pred_y = F.conv2d(predictions, sobel_y, padding=1)
        grad_target_x = F.conv2d(targets, sobel_x, padding=1)
        grad_target_y = F.conv2d(targets, sobel_y, padding=1)

        # Gradient magnitude loss
        grad_loss = F.mse_loss(grad_pred_x, grad_target_x) + F.mse_loss(
            grad_pred_y, grad_target_y
        )

        return grad_loss

    def _ssim_loss(self, predictions, targets):
        """
        Structural Similarity loss component[4]
        """
        # Window size for local comparison
        window_size = 11
        sigma = 1.5

        # Create Gaussian window
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        window = g.outer(g).unsqueeze(0).unsqueeze(0)

        if predictions.is_cuda:
            window = window.cuda()

        # Compute local means
        mu1 = F.conv2d(predictions, window, padding=window_size // 2, groups=1)
        mu2 = F.conv2d(targets, window, padding=window_size // 2, groups=1)

        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2

        # Compute local variances and covariance
        sigma1_sq = (
            F.conv2d(predictions**2, window, padding=window_size // 2, groups=1)
            - mu1_sq
        )
        sigma2_sq = (
            F.conv2d(targets**2, window, padding=window_size // 2, groups=1) - mu2_sq
        )
        sigma12 = (
            F.conv2d(predictions * targets, window, padding=window_size // 2, groups=1)
            - mu1_mu2
        )

        # SSIM constants
        c1 = 0.01**2
        c2 = 0.03**2

        # Compute SSIM
        ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )

        return 1 - torch.mean(ssim)


def create_continuous_similarity_mask(frame, ground_truth_bg, method="combined"):
    """
    Create continuous similarity measures instead of binary masks.
    Returns values between 0 (dissimilar) and 1 (very similar).
    """
    import cv2
    import numpy as np

    # Convert to different color spaces for robust comparison
    frame_lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(ground_truth_bg, cv2.COLOR_RGB2LAB).astype(np.float32)

    if method == "perceptual_distance":
        # Perceptual distance in LAB space (normalized)
        diff_lab = np.linalg.norm(frame_lab - bg_lab, axis=2)
        similarity = np.exp(-diff_lab / 50.0)  # Exponential decay

    elif method == "combined":
        # LAB distance component
        diff_lab = np.linalg.norm(frame_lab - bg_lab, axis=2)
        sim_lab = np.exp(-diff_lab / 40.0)

        # HSV distance component
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV).astype(np.float32)
        bg_hsv = cv2.cvtColor(ground_truth_bg, cv2.COLOR_RGB2HSV).astype(np.float32)
        diff_hsv = np.linalg.norm(frame_hsv[:, :, :2] - bg_hsv[:, :, :2], axis=2)
        sim_hsv = np.exp(-diff_hsv / 30.0)

        # Weighted combination
        similarity = 0.6 * sim_lab + 0.4 * sim_hsv

    # Apply soft morphological operations to reduce noise
    similarity = cv2.GaussianBlur(similarity.astype(np.float32), (5, 5), 1.0)

    return np.clip(similarity, 0, 1)


# %% [markdown]
# ### Model


# %%
class BackgroundLeakSegmenter(L.LightningModule):
    """Lightning module for background leak segmentation[6]"""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        loss_type: str = "combined",
        model_architecture: str = "unet",
        height: int = 720,
        width: int = 1280,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        # Now properly instantiated
        self.loss_fn = ContinuousRegressionLoss(
            loss_type=loss_type,
            alpha=1.0,  # Weight for MSE
            beta=0.5,  # Weight for gradient loss
            gamma=0.3,  # Weight for SSIM loss
        )

        # Build model
        self.model = self._build_model(model_architecture)

        # Metrics
        self.train_mae = []
        self.val_mae = []

    def _build_model(self, architecture: str):
        """Build segmentation model"""
        if architecture == "unet":
            return self._build_unet()
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _build_unet(self):
        """Lightweight U-Net for memory efficiency"""
        return nn.Sequential(
            # Encoder
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bottleneck
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            # Decoder
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            # Output
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        frames = batch["frame"]
        similarity_masks = batch["similarity_mask"]

        # Forward pass
        predictions = self.forward(frames)
        loss = self.loss_fn(predictions, similarity_masks)

        # Calculate MAE for monitoring
        mae = torch.mean(torch.abs(predictions - similarity_masks))
        self.train_mae.append(mae.item())

        # Log metrics[2]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae", mae, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        frames = batch["frame"]
        similarity_masks = batch["similarity_mask"]

        predictions = self.forward(frames)
        loss = self.loss_fn(predictions, similarity_masks)
        mae = torch.mean(torch.abs(predictions - similarity_masks))

        self.val_mae.append(mae.item())

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mae", mae, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,  # verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def test_step(self, batch, batch_idx):
        """
        Modified test step that reconstructs complete backgrounds from videos
        """
        # Get the video path from the batch (you'll need to modify dataset to include this)
        video_path = batch["video_path"][0]  # Assuming batch size 1 for testing
        background_path = batch["background_path"][0]
        mask_path = batch["mask_path"][0]

        height = self.height
        width = self.width

        # Reconstruct background from complete video
        reconstructed_bg = self.reconstruct_background_from_video(
            video_path, frame_size=(height, width)
        )

        # Load ground truth for evaluation
        ground_truth_bg = load_background(background_path)
        ground_truth_bg = cv2.resize(ground_truth_bg, (height, width))
        ground_truth_bg = cv2.cvtColor(ground_truth_bg, cv2.COLOR_BGR2RGB)

        evaluation_mask = load_mask(mask_path)
        evaluation_mask = cv2.resize(evaluation_mask, (height, width))

        # Calculate CIEDE2000 metric
        delta_e = metric_CIEDE2000(reconstructed_bg, ground_truth_bg, evaluation_mask)
        reconstruction_score = evaluate(delta_e, evaluation_mask)

        # Log results
        self.log("test_reconstruction_score", reconstruction_score, prog_bar=True)
        self.log(
            "test_delta_e_mean", float(np.mean(delta_e[evaluation_mask > 0]))
        )  # this is the mean delta E for pixels where mask is > 0

        return {
            "reconstruction_score": reconstruction_score,
            "reconstructed_bg": reconstructed_bg,
            "ground_truth_bg": ground_truth_bg,
        }

    # def reconstruct_background_from_video(self, video_path, frame_size=(720, 1280)):
    #     """
    #     Reconstruct background by processing all frames and keeping pixels with highest probability
    #     """
    #     self.eval()
    #     device = self.device

    #     cap = cv2.VideoCapture(str(video_path))
    #     if not cap.isOpened():
    #         raise ValueError(f"Cannot open video file {video_path}")

    #     if len(frame_size) == 2:
    #         height, width = frame_size
    #         cv2_size = (width, height)  # OpenCV uses (width, height)

    #     height, width = frame_size
    #     reconstruction_prob = np.zeros((height, width), dtype=np.float32)  # Store max probability per pixel
    #     reconstruction_img = np.zeros((height, width, 3), dtype=np.uint8)  # Store pixel values

    #     with torch.no_grad():
    #         frame_count = 0
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 break

    #             # Preprocess frame (SAME AS TRAINING)
    #             # frame_resized = cv2.resize(frame, frame_size)
    #             # frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    #             frame_rgb = preprocess_frame(frame, frame_size)

    #             # Convert to tensor and normalize (EXACTLY like in training dataset)
    #             # input_tensor = torch.FloatTensor(frame_rgb).permute(2, 0, 1) / 255.0
    #             # input_tensor = input_tensor.unsqueeze(0).to(device)  # Add batch dimension
    #             input_tensor = frame_to_tensor(frame_rgb).unsqueeze(0).to(device)

    #             # Predict
    #             pred = self(input_tensor).squeeze(0).squeeze(0).cpu().numpy()  # shape: (H, W) Outputs shape (W, H)

    #             # Update reconstruction where prediction probability is higher
    #             mask_update = pred > reconstruction_prob
    #             for c in range(3):
    #                 reconstruction_img[:, :, c][mask_update] = frame_rgb[:, :, c][mask_update]
    #             reconstruction_prob[mask_update] = pred[mask_update]

    #             frame_count += 1
    #             if frame_count % 100 == 0:  # Progress logging
    #                 print(f"Processed {frame_count} frames...")

    #     cap.release()
    #     print(f"Reconstruction complete. Processed {frame_count} total frames.")
    #     print(f"Coverage: {np.mean(reconstruction_prob > 0.1) * 100:.1f}% of pixels have prediction > 0.1")

    #     return reconstruction_img

    def reconstruct_background_from_video(self, video_path, frame_size=(720, 1280)):
        """
        Reconstruct background using consistent preprocessing
        """
        self.eval()
        device = self.device

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file {video_path}")

        # Fix: Handle frame_size consistently
        if len(frame_size) == 2:
            # Assume frame_size is (height, width), convert for cv2.resize
            height, width = frame_size
            cv2_size = (width, height)  # cv2.resize expects (width, height)

        # Initialize arrays with correct dimensions
        reconstruction_prob = np.zeros((height, width), dtype=np.float32)
        reconstruction_img = np.zeros((height, width, 3), dtype=np.uint8)

        with torch.no_grad():
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Use consistent preprocessing with proper cv2.resize format
                frame_resized = cv2.resize(frame, cv2_size)  # Use (width, height)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                input_tensor = frame_to_tensor(frame_rgb).unsqueeze(0).to(device)

                # Predict
                pred = self(input_tensor).squeeze(0).squeeze(0).cpu().numpy()

                # Debug: Print shapes to verify consistency
                if frame_count == 0:
                    print(f"Frame RGB shape: {frame_rgb.shape}")
                    print(f"Prediction shape: {pred.shape}")
                    print(f"Reconstruction prob shape: {reconstruction_prob.shape}")

                # Update reconstruction where prediction probability is higher
                mask_update = pred > reconstruction_prob
                for c in range(3):
                    reconstruction_img[:, :, c][mask_update] = frame_rgb[:, :, c][
                        mask_update
                    ]
                reconstruction_prob[mask_update] = pred[mask_update]

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames...")

        cap.release()
        return reconstruction_img


# %% [markdown]
# ### Training

# %%


def train_model():
    """Main training function with memory optimization"""

    # Initialize data module
    data_module = BackgroundLeakDataModule(
        backgrounds_dir="data/public/backgrounds",
        videos_dir="data/public/videos",
        masks_dir="data/public/masks",
        batch_size=4,  # Small batch size for memory efficiency
        num_workers=8,  # Adjusted for my laptop
        frames_per_video=50,  # Limit frames per video
    )

    # Initialize model
    model = BackgroundLeakSegmenter(learning_rate=1e-4, loss_type="combined")

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

    # Train model
    trainer.fit(model, data_module)

    # Test model
    trainer.test(model, data_module, ckpt_path="best")


if __name__ == "__main__":
    # Set memory optimization flags
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes

    # Enable garbage collection
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    train_model()

# %% [markdown]
# Testing

# %%
