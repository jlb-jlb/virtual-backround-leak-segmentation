import torch
import torch.nn as nn
import torch.nn.functional as F


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
