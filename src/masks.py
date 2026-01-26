def create_continuous_similarity_mask(
    frame, ground_truth_bg, method="combined", threshold: float = 0.0
):
    """
    Create continuous similarity measures instead of binary masks.
    Returns values between 0 (dissimilar) and 1 (very similar).
    Args:
        threshold: float in [0,1], If threshold > 0.0, returns binary mask.
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
    # If threshold is set, return binary mask
    if threshold > 0.0:
        binary_mask = (similarity >= threshold).astype(np.float32)
        return binary_mask
    # Otherwise return continuous similarity
    return np.clip(similarity, 0, 1)
