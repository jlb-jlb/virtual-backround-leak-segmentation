from pathlib import Path
from typing import List, Tuple
from matplotlib.pyplot import imshow, subplots
import matplotlib.pyplot as plt


def display_video(
    video: dict, sample: Tuple[str | Path, str | Path, str | Path]
) -> None:
    """_summary_

    Args:
        video (dict): _description_
        sample (Tuple[str, str, str]): _description_
    """

    fig, ax = subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(video["video"][92])
    ax[1].imshow(video["mask"])
    ax[2].imshow(video["background"])
    ax[0].set_title(f"Video Frame: {sample[0]}")
    ax[1].set_title(f"Mask: {sample[1]}")
    ax[2].set_title(f"Background: {sample[2]}")
    ax[0].axis("off")
    ax[1].axis("off")
    ax[2].axis("off")
    plt.tight_layout()
    plt.show()


def load_triplets(
    backgrounds_dir: str | Path,
    videos_dir: str | Path,
    masks_dir: str | Path,
) -> List[Tuple[Path, Path, Path]]:
    """
    Collect (video, mask, background) filepath triplets that belong together.

    The expected filename patterns are

    * videos:  <person>_<gt_bg>_<virt_bg>_mp.mp4
               e.g. 2_i_kitchen_bridge_mp.mp4
    * masks:   <person>_<gt_bg>_mp.png
               e.g. 2_i_kitchen_mp.png
    * backgrounds: <gt_bg>.png
               e.g. kitchen.png

    Parameters
    ----------
    backgrounds_dir, videos_dir, masks_dir
        Directory paths (str or pathlib.Path).

    Returns
    -------
    list[tuple[pathlib.Path, pathlib.Path, pathlib.Path]]
        Matched (video, mask, background) filepaths.

    Notes
    -----
    * Only videos for which **both** the corresponding mask **and**
      background image exist are returned.
    * You can freely switch to using `os.path` instead of `pathlib.Path`
      if you prefer.
    """
    backgrounds_dir = Path(backgrounds_dir)
    videos_dir = Path(videos_dir)
    masks_dir = Path(masks_dir)

    # Build quick look-up tables for masks and backgrounds
    masks = {p.name: p for p in masks_dir.glob("*.png")}
    bgs = {p.stem: p for p in backgrounds_dir.glob("*.png")}

    triplets: List[Tuple[Path, Path, Path]] = []

    for video_path in videos_dir.glob("*.mp4"):
        stem_parts = video_path.stem.split("_")
        if len(stem_parts) < 5 or stem_parts[-1] != "mp":
            # Skip unexpected file names
            continue

        # Extract pieces
        person = "_".join(stem_parts[:2])  # e.g. "2_i"
        gt_bg = stem_parts[2]  # e.g. "kitchen"
        # virtual_bg = stem_parts[3]  # not needed here

        # Expected companion names
        mask_name = f"{person}_{gt_bg}_mp.png"
        bg_path = bgs.get(gt_bg)

        # Check presence
        mask_path = masks.get(mask_name)
        if mask_path is None or bg_path is None:
            # Incomplete set → ignore or optionally warn
            continue

        triplets.append((video_path, mask_path, bg_path))

    return triplets
