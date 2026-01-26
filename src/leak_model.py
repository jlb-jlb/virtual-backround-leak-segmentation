from pathlib import Path
import torch
import torch.nn as nn
import cv2
import numpy as np
import lightning as L
import os

from tqdm import tqdm

from .util import load_background, load_mask, metric_CIEDE2000, evaluate  # type: ignore
from .losses import ContinuousRegressionLoss
from .video_dataset import frame_to_tensor, preprocess_frame, edge_filter, dog_filter, mode_image


# Add a simple convolutional network option
class ConvNetSimple(nn.Module):
    def __init__(self, channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


# Add standalone U-Net module for modularity
class UNetSimple(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            # Encoder
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Bottleneck
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Decoder
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Output
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class BackgroundLeakSegmenter(L.LightningModule):
    """Lightning module for background leak segmentation[6]"""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        loss_type: str = "combined",
        model_architecture: str = "unet",  # 'unet' or 'convnet_simple'
        in_channels: int = 3,
        height: int = 720,
        width: int = 1280,
        use_edge: bool = False,
        use_dog: bool = False,
        use_mode: bool = False,
        mode_dir: str = None,  # type: ignore
    ):
        super().__init__()
        # Save hyperparameters including in_channels
        self.save_hyperparameters()
        # Model config
        self.model_architecture = model_architecture
        self.in_channels = in_channels
        self.height = height
        self.width = width
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

        # Filter flags for test-time reconstruction
        self.use_edge = use_edge
        self.use_dog = use_dog
        self.use_mode = use_mode
        # Directory for storing/loading mode images
        self.mode_dir = mode_dir or os.path.join(os.getcwd(), "mode_images")
        if self.use_mode:
            os.makedirs(self.mode_dir, exist_ok=True)

    def _build_model(self, architecture: str):
        """Build segmentation model: 'unet' or 'convnet_simple'"""
        if architecture == "unet":
            # Use modular UNetSimple with adjustable input channels
            return UNetSimple(in_channels=self.in_channels)
        elif architecture == "convnet_simple":
            # Simple convolutional network with adjustable input channels
            return ConvNetSimple(channels=self.in_channels)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _build_unet(self):
        """Return modular UNetSimple instance"""
        return UNetSimple(in_channels=self.in_channels)

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
        """Modified test step that reconstructs complete backgrounds from videos"""
        video_path = batch["video_path"][0]
        background_path = batch["background_path"][0]
        mask_path = batch["mask_path"][0]

        print(f"Processing video: {Path(video_path).name}")

        # Use consistent frame_size as (width, height)
        frame_size = (1280, 720)  # (width, height)
        width, height = frame_size

        # Reconstruct background from complete video
        reconstructed_bg = self.reconstruct_background_from_video(
            video_path, frame_size=frame_size
        )

        # Load ground truth with consistent sizing
        ground_truth_bg = load_background(background_path)
        ground_truth_bg = preprocess_frame(ground_truth_bg, frame_size)

        # Load evaluation mask with consistent sizing
        evaluation_mask = load_mask(mask_path)
        evaluation_mask = cv2.resize(evaluation_mask, frame_size)

        # Calculate CIEDE2000 metric
        delta_e = metric_CIEDE2000(reconstructed_bg, ground_truth_bg, evaluation_mask)
        reconstruction_score = evaluate(delta_e, evaluation_mask)

        # === SAVE IMAGES ===
        self.save_reconstruction_results(
            video_path,
            reconstructed_bg,
            ground_truth_bg,
            evaluation_mask,
            delta_e,
            reconstruction_score,
        )

        # Log results with explicit batch_size
        self.log(
            "test_reconstruction_score",
            reconstruction_score,
            prog_bar=True,
            batch_size=1,
        )
        self.log("test_delta_e_mean", np.mean(delta_e[evaluation_mask > 0]), batch_size=1)  # type: ignore

        print(f"Reconstruction score: {reconstruction_score:.4f}")

        return {
            "reconstruction_score": reconstruction_score,
            "video_name": Path(video_path).name,
        }

    def _compute_mode_image(self, video_path, frame_size):
        """Compute per-video mode RGB image using PyTorch over all frames."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            rgb = preprocess_frame(frm, frame_size)
            frames.append(frame_to_tensor(rgb))
        cap.release()
        if frames:
            stack = torch.stack(frames)  # (F, C, H, W)
            mode_tensor, _ = torch.mode(stack, dim=0)
            mode_np = (mode_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
        else:
            h, w = frame_size[1], frame_size[0]
            mode_np = np.zeros((h, w, 3), dtype=np.uint8)
        return mode_np

    def reconstruct_background_from_video(self, video_path, frame_size=(1280, 720)):
        """
        Reconstruct background using consistent preprocessing
        Args:
            frame_size: (width, height) - cv2 convention
        """
        self.eval()
        device = self.device

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file {video_path}")

        width, height = frame_size

        # Initialize arrays with numpy convention (height, width)
        reconstruction_prob = np.zeros((height, width), dtype=np.float32)
        reconstruction_img = np.zeros((height, width, 3), dtype=np.uint8)

        # Precompute mode image if enabled
        if self.use_mode:
            mode_np = self._compute_mode_image(video_path, frame_size)
            mode_tensor = frame_to_tensor(mode_np)

        with torch.no_grad():
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Use consistent preprocessing - returns (height, width, 3)
                frame_rgb = preprocess_frame(frame, frame_size)
                # Build input tensor with optional filter channels
                base = frame_to_tensor(frame_rgb)
                channels = [base]
                if self.use_edge:
                    edges = edge_filter(frame_rgb)
                    channels.append(torch.FloatTensor(edges).unsqueeze(0) / 255.0)
                if self.use_dog:
                    dog = dog_filter(frame_rgb)
                    channels.append(torch.FloatTensor(dog).unsqueeze(0) / 255.0)
                # Append mode channel if enabled
                if self.use_mode:
                    channels.append(mode_tensor)
                input_tensor = torch.cat(channels, dim=0).unsqueeze(0).to(device)

                # Predict
                pred = self(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
                # pred shape: (height, width)

                # Debug: Print shapes to verify consistency
                if frame_count == 0:
                    print(f"Frame RGB shape: {frame_rgb.shape}")
                    print(f"Prediction shape: {pred.shape}")
                    print(f"Reconstruction prob shape: {reconstruction_prob.shape}")
                    print(f"Reconstruction img shape: {reconstruction_img.shape}")

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
        print(f"Reconstruction complete. Processed {frame_count} total frames.")
        print(
            f"Coverage: {np.mean(reconstruction_prob > 0.1) * 100:.1f}% of pixels have prediction > 0.1"
        )

        return reconstruction_img  # Shape: (height, width, 3)

    def save_reconstruction_results(
        self,
        video_path,
        reconstructed_bg,
        ground_truth_bg,
        evaluation_mask,
        delta_e,
        reconstruction_score,
    ):
        """Save reconstruction results to disk"""

        # Create output directory
        output_dir = Path("reconstruction_results")
        output_dir.mkdir(exist_ok=True)

        # Extract meaningful filename from video path
        video_name = Path(video_path).stem  # e.g., "2_i_kitchen_bridge_mp"

        # Save reconstructed background
        reconstructed_bgr = cv2.cvtColor(reconstructed_bg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(output_dir / f"{video_name}_reconstructed.png"), reconstructed_bgr
        )

        # Save ground truth for comparison
        ground_truth_bgr = cv2.cvtColor(ground_truth_bg, cv2.COLOR_RGB2BGR)
        cv2.imwrite(
            str(output_dir / f"{video_name}_ground_truth.png"), ground_truth_bgr
        )

        # Save evaluation mask
        cv2.imwrite(str(output_dir / f"{video_name}_mask.png"), evaluation_mask)

        # Create and save difference visualization
        diff_visualization = self.create_difference_visualization(
            reconstructed_bg, ground_truth_bg, evaluation_mask, delta_e
        )
        cv2.imwrite(
            str(output_dir / f"{video_name}_difference.png"), diff_visualization
        )

        # Save metadata
        metadata = {
            "video_name": Path(video_path).name,
            "reconstruction_score": float(reconstruction_score),
            "mean_delta_e": float(np.mean(delta_e[evaluation_mask > 0])),
            "coverage_10": float(np.mean(delta_e[evaluation_mask > 0] < 10))
            * 100,  # % pixels with delta_e < 10
            "coverage_4": float(np.mean(delta_e[evaluation_mask > 0] < 4))
            * 100,  # % pixels with delta_e < 4
        }

        import json

        with open(output_dir / f"{video_name}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved results for {video_name} to {output_dir}")

    def create_difference_visualization(
        self, reconstructed_bg, ground_truth_bg, evaluation_mask, delta_e
    ):
        """Create a visualization showing reconstruction quality"""

        # Create side-by-side comparison
        height, width = reconstructed_bg.shape[:2]
        comparison = np.zeros((height, width * 3, 3), dtype=np.uint8)

        # Left: Reconstructed
        comparison[:, :width] = reconstructed_bg

        # Middle: Ground Truth
        comparison[:, width : 2 * width] = ground_truth_bg

        # Right: Difference heatmap
        # Normalize delta_e to 0-255 range for visualization
        delta_e_normalized = np.clip(delta_e / 20.0 * 255, 0, 255).astype(np.uint8)

        # Apply colormap (red = high difference, green = low difference)
        heatmap = cv2.applyColorMap(delta_e_normalized, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Fix: Apply mask correctly to each channel
        mask_2d = evaluation_mask > 0  # Boolean mask (height, width)

        # Set invalid areas to gray - do this per channel
        for c in range(3):
            heatmap[:, :, c][~mask_2d] = 50  # Gray for invalid areas

        comparison[:, 2 * width :] = heatmap

        # Add text labels
        comparison = cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR)
        cv2.putText(
            comparison,
            "Reconstructed",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            "Ground Truth",
            (width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            comparison,
            "Difference",
            (2 * width + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        return comparison

    # Inference
    def predict_step(self, batch, batch_idx):
        """
        Lightning prediction step for batch processing
        Args:
            batch: Dictionary containing 'frames' key with tensor of shape (B, F, 3, H, W)
                where B=batch_size, F=num_frames, 3=channels, H=height, W=width
        Returns:
            Reconstructed background as tensor
        """
        frames = batch["frames"]  # Shape: (B, F, 3, H, W)

        # Process each batch item
        results = []
        for b in range(frames.size(0)):
            frame_sequence = frames[b]  # Shape: (F, 3, H, W)
            reconstructed = self.reconstruct_from_tensor_frames(frame_sequence)
            results.append(reconstructed)

        return torch.stack(results)

    def reconstruct_from_numpy_frames(
        self, frames_array, frame_size=(1280, 720), verbose: bool = False
    ):
        """
        Reconstruct background from numpy array of frames

        Args:
            frames_array: numpy array of shape (num_frames, height, width, 3) in BGR format
            frame_size: (width, height) tuple for output size

        Returns:
            reconstructed_bg: numpy array of shape (height, width, 3) in BGR format
        """
        self.eval()
        device = self.device
        num_frames, original_height, original_width, _ = frames_array.shape
        width, height = frame_size

        # Initialize reconstruction arrays
        reconstruction_prob = np.zeros((height, width), dtype=np.float32)
        reconstruction_img = np.zeros((height, width, 3), dtype=np.uint8)
        if verbose:
            print(f"Processing {num_frames} frames for reconstruction...")

        # Precompute mode image if enabled
        if self.use_mode:
            mode_tensor = torch.mode(torch.from_numpy(frames_array), dim=0)[0].permute(2, 0, 1)
            # scale mode tensor to [0, 1] range
            mode_tensor = mode_tensor.float().squeeze(0) / 255.0
            # print(f"Mode tensor shape: {mode_tensor.shape}")
            # mode_tensor = frame_to_tensor(mode_img)



        with torch.no_grad():
            iterator = range(num_frames)
            if verbose:
                iterator = tqdm(iterator)
            for frame_idx in iterator:
                frame_bgr = frames_array[frame_idx]
                # Convert BGR->RGB and resize
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (width, height))
                # Build input tensor with optional filters
                base = frame_to_tensor(frame_rgb)
                # print(f"Base tensor shape: {base.shape}")
                channels = [base]
                if self.use_edge:
                    edges = edge_filter(frame_rgb)
                    channels.append(torch.FloatTensor(edges).unsqueeze(0) / 255.0)
                if self.use_dog:
                    dog = dog_filter(frame_rgb)
                    channels.append(torch.FloatTensor(dog).unsqueeze(0) / 255.0)
                # Append mode channel if enabled
                if self.use_mode:
                    channels.append(mode_tensor)
                input_tensor = torch.cat(channels, dim=0).unsqueeze(0).to(device)

                pred = self(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
                # Update reconstruction where prediction is higher
                mask_update = pred > reconstruction_prob
                for c in range(3):
                    reconstruction_img[:, :, c][mask_update] = frame_rgb[:, :, c][
                        mask_update
                    ]
                reconstruction_prob[mask_update] = pred[mask_update]
                # if verbose and (frame_idx + 1) % 100 == 0:
                #     print(f"Processed {frame_idx + 1}/{num_frames} frames...")

        if verbose:
            print(
                f"Reconstruction complete. Coverage: {np.mean(reconstruction_prob > 0.1) * 100:.1f}% pixels"
            )
        # Convert back to BGR
        reconstructed_bgr = cv2.cvtColor(reconstruction_img, cv2.COLOR_RGB2BGR)
        
        return reconstructed_bgr  # Shape: (height, width, 3) in BGR format

    def reconstruct_from_tensor_frames(self, frames_tensor):
        """
        Reconstruct background from tensor frames (for Lightning predict_step)

        Args:
            frames_tensor: torch tensor of shape (num_frames, 3, height, width)

        Returns:
            reconstructed_bg: torch tensor of shape (3, height, width)
        """
        # Prepare for per-frame inference
        self.eval()
        device = self.device
        num_frames, _, height, width = frames_tensor.shape
        frames_tensor = frames_tensor.to(device)

        # Compute mode tensor if enabled
        if self.use_mode:
            # mode over RGB channels only
            mode_rgb = torch.mode(frames_tensor[:, :3, :, :], dim=0)[0]  # (3, H, W)
            # Keep on device
            mode_tensor = mode_rgb

        # Initialize reconstruction buffers
        reconstruction_prob = torch.zeros((height, width), device=device)
        reconstruction_img = torch.zeros((3, height, width), device=device)

        with torch.no_grad():
            for idx in range(num_frames):
                # Build input channels
                base = frames_tensor[idx, :3]  # RGB
                channels = [base]
                if self.use_edge or self.use_dog:
                    # convert to numpy for filters
                    np_rgb = (base.permute(1, 2, 0).cpu().numpy() * 255).astype(
                        np.uint8
                    )
                if self.use_edge:
                    edges = edge_filter(np_rgb)
                    channels.append(
                        torch.FloatTensor(edges).unsqueeze(0).to(device) / 255.0
                    )
                if self.use_dog:
                    dog = dog_filter(np_rgb)
                    channels.append(
                        torch.FloatTensor(dog).unsqueeze(0).to(device) / 255.0
                    )
                if self.use_mode:
                    channels.append(mode_tensor)
                inp = torch.cat(channels, dim=0).unsqueeze(0)
                # Predict mask
                pred = self(inp).squeeze(0).squeeze(0)  # (H, W)
                # Update reconstruction
                mask_u = pred > reconstruction_prob
                # Update each RGB channel
                for c in range(3):
                    reconstruction_img[c][mask_u] = base[c][mask_u]
                reconstruction_prob[mask_u] = pred[mask_u]

        return reconstruction_img

    @classmethod
    def load_for_inference(cls, checkpoint_path):
        """
        Load model for inference (convenience method)

        Args:
            checkpoint_path: Path to the checkpoint file

        Returns:
            model: Loaded model in eval mode
        """
        model = cls.load_from_checkpoint(checkpoint_path)
        model.eval()
        return model
