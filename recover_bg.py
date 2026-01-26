# recover_bg.py
from pathlib import Path
import numpy as np
import cv2
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import colour
from util import load_video, load_mask, load_background, metric_CIEDE2000, evaluate
from src.data_loader import load_triplets  # <- your earlier helper

# -------------------------- hyper-parameters --------------------------
MIN_FRAMES = 3  # ≥ this many leaks required
DELTA_CEILAB = 6.0  # component-mean separation threshold (ΔE₀₀)
INPAINT_RAD = 3  # OpenCV inpaint radius
# ----------------------------------------------------------------------


def recover_background(video_p: Path, mask_p: Path) -> np.ndarray:
    frames = load_video(video_p)  # list[H,W,3] BGR
    mask = load_mask(mask_p).astype(bool)  # person = True/255
    frames = np.stack(frames, axis=0)  # (T,H,W,3)
    frames = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2LAB) for f in frames],  # iterate over list
        axis=0,
    )

    T, H, W, _ = frames.shape
    recovered = np.full((H, W, 3), np.nan, dtype=np.float32)
    bg_indices = ~mask

    # reshape to (N,T,3) where N = H*W
    pix = frames[:, bg_indices].transpose(1, 0, 2)

    # fit 2-GMM per pixel
    for idx, series in enumerate(pix):
        gmm = GaussianMixture(n_components=2, covariance_type="diag", max_iter=50)
        gmm.fit(series)

        # sort by component weight
        order = np.argsort(gmm.weights_)[::-1]  # 0 = dominant
        means = gmm.means_[order]
        counts = gmm.weights_[order] * T

        # rare component test
        if counts[1] >= MIN_FRAMES:
            delta_e = colour.delta_E(means[0][None], means[1][None], method="CIE 2000")[
                0
            ]
            if delta_e >= DELTA_CEILAB:
                i, j = np.argwhere(bg_indices)[idx]
                recovered[i, j] = means[1]  # Lab value

    # post-processing
    recovered_bgr = cv2.cvtColor(
        np.nan_to_num(recovered, nan=0).astype(np.uint8), cv2.COLOR_Lab2BGR
    )
    holes = np.isnan(recovered[..., 0]).astype(np.uint8)
    recovered_bgr = cv2.inpaint(recovered_bgr, holes, INPAINT_RAD, cv2.INPAINT_TELEA)

    return recovered_bgr


# ---------------------- demo / quick evaluation -----------------------
if __name__ == "__main__":
    triplets = load_triplets(
        "data/public/backgrounds", "data/public/videos", "data/public/masks"
    )

    for vid_p, mask_p, gt_bg_p in tqdm(triplets):
        rec = recover_background(vid_p, mask_p)
        gt_bg = load_background(gt_bg_p)

        # optional evaluation
        delta = metric_CIEDE2000(rec, gt_bg, load_mask(mask_p))
        score = evaluate(delta, load_mask(mask_p))
        print(f"{vid_p.name}: reconstruction score = {score:.3f}")
