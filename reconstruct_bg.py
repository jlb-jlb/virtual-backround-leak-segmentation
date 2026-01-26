import numpy as np
import cv2
from sklearn.cluster import KMeans
from pathlib import Path
from typing import List
from tqdm import tqdm

from util import load_video


# --- Core reconstruction function ---
def reconstruct_background_via_clustering(
    video_frames_list: List[np.ndarray],
    n_clusters: int = 2,
    min_valid_leak_cluster_size: int = 3,
) -> np.ndarray:
    """
    Reconstructs the background by clustering pixel color time series.

    Args:
        video_frames_list (List[np.ndarray]): A list of video frames (H, W, 3) as NumPy arrays.
        n_clusters (int): The number of clusters (k) for K-Means. Default is 2.
        min_valid_leak_cluster_size (int): Minimum number of frames a "leak" cluster
                                           must have to be considered valid. Default is 3.

    Returns:
        np.ndarray: The reconstructed background image (H, W, 3).
    """
    if not video_frames_list:
        raise ValueError("Input video_frames_list is empty.")

    num_frames = len(video_frames_list)
    if num_frames == 0:
        raise ValueError("Video has no frames.")

    first_frame = video_frames_list[0]
    height, width, channels = first_frame.shape

    # Convert list of frames to a single NumPy array for efficient slicing
    # video_data shape: (num_frames, height, width, channels)
    video_data = np.array(video_frames_list)

    reconstructed_image = np.zeros((height, width, channels), dtype=np.uint8)

    print(
        f"Starting background reconstruction for an image of size {width}x{height} using {num_frames} frames."
    )
    print(
        f"Clustering with k={n_clusters}. Min valid leak cluster size: {min_valid_leak_cluster_size}."
    )

    for r in tqdm(range(height)):
        # Print progress every 10 rows or so for larger images
        # if r % max(1, height // 20) == 0 or r == height -1 :
        # print(f"Processing row {r + 1}/{height}...")
        for c in range(width):
            pixel_time_series = video_data[:, r, c, :]  # Shape: (num_frames, channels)

            # Get unique colors and their counts for this pixel's time series
            unique_colors, counts = np.unique(
                pixel_time_series, axis=0, return_counts=True
            )

            if len(unique_colors) < n_clusters:
                # If not enough unique colors to form n_clusters,
                # assign the most frequent color (mode) or the single unique color.
                if len(unique_colors) > 0:
                    most_frequent_color_idx = np.argmax(counts)
                    reconstructed_image[r, c] = unique_colors[
                        most_frequent_color_idx
                    ].astype(np.uint8)
                # else: pixel remains black (or initial value of reconstructed_image)
                continue

            try:
                # Using n_init='auto' which is default in newer scikit-learn and picks 1 for lloyd/elkan.
                # For very small datasets per pixel, multiple initializations might be overkill or slow.
                # random_state ensures reproducibility for each pixel's clustering.
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
                kmeans.fit(pixel_time_series)
            except Exception as e:
                # This might happen for some degenerate cases in pixel data with KMeans
                print(
                    f"  Warning: KMeans failed for pixel ({r},{c}): {e}. Assigning most frequent color."
                )
                if len(unique_colors) > 0:  # Should always be true here
                    most_frequent_color_idx = np.argmax(counts)
                    reconstructed_image[r, c] = unique_colors[
                        most_frequent_color_idx
                    ].astype(np.uint8)
                continue

            labels = kmeans.labels_
            centroids = kmeans.cluster_centers_  # Float values

            # Ensure cluster_sizes array has length n_clusters, even if some clusters are empty
            # (though KMeans typically doesn't produce empty clusters with standard data)
            cluster_sizes = np.bincount(labels, minlength=n_clusters)

            # Find the smallest and largest cluster indices
            # Handle cases where all clusters might be empty (highly unlikely with KMeans fit)
            if not np.any(cluster_sizes > 0):  # All clusters are empty
                if len(unique_colors) > 0:  # Fallback to most frequent
                    most_frequent_color_idx = np.argmax(counts)
                    reconstructed_image[r, c] = unique_colors[
                        most_frequent_color_idx
                    ].astype(np.uint8)
                continue  # Should not be reachable if KMeans worked and unique_colors > 0

            smallest_cluster_idx = np.argmin(cluster_sizes)
            largest_cluster_idx = np.argmax(cluster_sizes)  # Fallback for virtual BG

            # Decide which centroid to use based on the "leak" hypothesis
            chosen_centroid = None
            if cluster_sizes[smallest_cluster_idx] >= min_valid_leak_cluster_size:
                # The smallest cluster is considered a valid leak
                chosen_centroid = centroids[smallest_cluster_idx]
            else:
                # The smallest cluster is too small (likely noise or no confident leak).
                # Fallback to the largest cluster (assumed to be the virtual background).
                chosen_centroid = centroids[largest_cluster_idx]

            reconstructed_image[r, c] = chosen_centroid.astype(np.uint8)

    print("Background reconstruction finished.")
    return reconstructed_image


# --- Main execution block ---
if __name__ == "__main__":
    # --- Configuration ---
    # !!! IMPORTANT: Replace this with the actual path to your video file !!!
    video_file_path_str = "data/public/videos/24_i_livingroom_leaves_mp.mp4"  # e.g., "path/to/your/videos/2_i_kitchen_bridge_mp.mp4"
    output_image_path_str = "reconstructed_background.png"

    # Clustering parameters
    NUM_KMEANS_CLUSTERS = 2  # k=2 for (virtual_bg, real_bg_leak)
    MIN_LEAK_CLUSTER_SIZE = 3  # Minimum frames for a leak to be considered valid

    # 1. Load the video frames
    print(f"Loading video from: {video_file_path_str}")
    video_frames = load_video(Path(video_file_path_str))

    if not video_frames:
        print("No frames loaded. Exiting.")
    else:
        print(f"Successfully loaded {len(video_frames)} frames.")
        # For very long videos or large frames, you might want to resize or process a subset of frames
        # Example: video_frames = video_frames[::2] # Process every 2nd frame
        # Example: video_frames = [cv2.resize(f, (f.shape[1]//2, f.shape[0]//2)) for f in video_frames]

        # 2. Reconstruct the background
        reconstructed_bg = reconstruct_background_via_clustering(
            video_frames,
            n_clusters=NUM_KMEANS_CLUSTERS,
            min_valid_leak_cluster_size=MIN_LEAK_CLUSTER_SIZE,
        )

        # 3. Display and Save the result
        cv2.imshow("Reconstructed Background", reconstructed_bg)
        print(
            f"\nReconstructed background generated. Press any key to close the display window."
        )

        # Save the image
        try:
            cv2.imwrite(output_image_path_str, reconstructed_bg)
            print(f"Reconstructed background saved to: {output_image_path_str}")
        except Exception as e:
            print(f"Error saving image: {e}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
