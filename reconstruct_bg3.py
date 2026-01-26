import numpy as np
import cv2
from pathlib import Path
from typing import List

# Ensure you have colour-science installed: pip install colour-science
try:
    import colour
except ImportError:
    print(
        "Error: colour-science library not found. Please install it: pip install colour-science"
    )
    print("Continuing without CIEDE2000, using simple RGB difference (less accurate).")
    colour = None

# Assuming util.py is in the same directory or Python path
# It should contain: load_video, metric_CIEDE2000, evaluate, load_mask, load_background
try:
    from util import load_video, metric_CIEDE2000, evaluate, load_mask, load_background
except ImportError:
    print("Error: util.py not found or missing necessary functions.")
    print(
        "Please ensure util.py with load_video, metric_CIEDE2000, evaluate, load_mask, load_background is available."
    )
    # Define a dummy load_video if util.py is not found, so the rest of the script can be parsed
    # This is just for preventing immediate script crash if util is missing; actual execution will fail.
    if "load_video" not in globals():

        def load_video(video_fpath: Path) -> list:
            print(
                f"Dummy load_video called for {video_fpath}. No frames will be loaded."
            )
            print("Please ensure util.py is correctly set up.")
            return []

        # Define dummy evaluation functions too, so script can be checked without full execution
        def load_mask(p):
            return np.array([[]], dtype=np.uint8)

        def load_background(p):
            return np.array([[[0, 0, 0]]], dtype=np.uint8)

        def metric_CIEDE2000(r, b, m):
            return np.array([[]], dtype=float)

        def evaluate(d, m):
            return 0.0


# --- Helper function to get mode color ---
def get_mode_rgb_color(pixel_time_series_data: np.ndarray) -> np.ndarray:
    """Calculates the mode (most frequent) color for a pixel's time series or a list of colors."""
    if (
        not isinstance(pixel_time_series_data, np.ndarray)
        or pixel_time_series_data.ndim == 0
    ):
        return np.array(
            [0, 0, 0], dtype=np.uint8
        )  # Should not happen with proper input
    if (
        pixel_time_series_data.ndim == 1
        and pixel_time_series_data.shape[0] % 3 == 0
        and pixel_time_series_data.shape[0] > 0
    ):  # A single color passed
        return pixel_time_series_data.astype(np.uint8)
    if pixel_time_series_data.shape[0] == 0:
        return np.array([0, 0, 0], dtype=np.uint8)  # Fallback for empty

    # Ensure it's a 2D array (list of colors)
    if (
        pixel_time_series_data.ndim == 1 and pixel_time_series_data.shape[0] == 3
    ):  # Single color passed as 1D array
        return pixel_time_series_data.astype(np.uint8)
    elif (
        pixel_time_series_data.ndim != 2 or pixel_time_series_data.shape[1] != 3
    ):  # Incorrect shape
        # print(f"Warning: Unexpected shape in get_mode_rgb_color: {pixel_time_series_data.shape}")
        if (
            pixel_time_series_data.size > 0
        ):  # Try to reshape if it's a flattened list of colors
            if pixel_time_series_data.size % 3 == 0:
                try:
                    pixel_time_series_data = pixel_time_series_data.reshape(-1, 3)
                except ValueError:
                    return np.array([0, 0, 0], dtype=np.uint8)
            else:  # Cannot determine colors
                return np.array([0, 0, 0], dtype=np.uint8)
        else:  # Empty
            return np.array([0, 0, 0], dtype=np.uint8)

    unique_colors, counts = np.unique(
        pixel_time_series_data, axis=0, return_counts=True
    )
    if len(unique_colors) > 0:
        return unique_colors[np.argmax(counts)].astype(np.uint8)
    return np.array([0, 0, 0], dtype=np.uint8)  # Fallback


# --- Color Difference Functions ---
def rgb_to_lab_for_colour_lib(rgb_color_uint8: np.ndarray) -> np.ndarray:
    """Converts a single RGB uint8 color to Lab float32 for colour-science."""
    rgb_float = rgb_color_uint8.astype(np.float32) / 255.0
    xyz_color = colour.sRGB_to_XYZ(rgb_float)
    lab_color = colour.XYZ_to_Lab(xyz_color)
    return lab_color.astype(np.float32)


def calculate_color_difference(color1_rgb_uint8, color2_rgb_uint8, method="CIEDE2000"):
    if colour and method == "CIEDE2000":
        color1_lab = rgb_to_lab_for_colour_lib(color1_rgb_uint8)
        color2_lab = rgb_to_lab_for_colour_lib(color2_rgb_uint8)
        return colour.delta_E(color1_lab, color2_lab, method="CIE 2000")
    else:
        return np.sum(
            np.abs(
                color1_rgb_uint8.astype(np.int16) - color2_rgb_uint8.astype(np.int16)
            )
        )


# --- Core reconstruction function ---
def reconstruct_background_median_baseline_transient(
    video_frames_list: List[np.ndarray],
    delta_e_deviation_threshold: float = 15.0,
    delta_e_consistency_threshold: float = 10.0,
    min_leak_duration_frames: int = 1,
    max_leak_duration_frames: int = 7,
    use_ciede2000: bool = True,
) -> np.ndarray:
    if not video_frames_list:
        raise ValueError("Input video_frames_list is empty.")
    num_frames = len(video_frames_list)
    if num_frames == 0:
        raise ValueError("Video has no frames.")

    video_data = np.array(video_frames_list)
    height, width, channels = video_data.shape[1:]  # Get dimensions from data

    diff_method = "CIEDE2000" if (colour and use_ciede2000) else "RGB_SIMPLE"
    print(
        f"Starting transient detection with median baseline. Image: {width}x{height}, Frames: {num_frames}. Diff method: {diff_method}"
    )
    print(
        f"Params: DevThresh={delta_e_deviation_threshold}, ConsThresh={delta_e_consistency_threshold}, LeakDur=({min_leak_duration_frames}-{max_leak_duration_frames})"
    )

    print("Calculating median frame as baseline...")
    median_frame_rgb = np.median(video_data, axis=0).astype(np.uint8)
    print("Median frame calculated.")

    reconstructed_image = np.zeros_like(median_frame_rgb, dtype=np.uint8)

    for r in range(height):
        if r % max(1, height // 20) == 0 or r == height - 1:
            print(f"Processing row {r + 1}/{height}...")
        for c in range(width):
            pixel_time_series = video_data[
                :, r, c, :
            ]  # Used for iterating through actual frame colors
            baseline_pixel_color_rgb = median_frame_rgb[
                r, c
            ]  # Baseline from median frame

            collected_valid_leak_colors_rgb = []
            frame_idx = 0
            while frame_idx < num_frames:
                current_frame_color_rgb = pixel_time_series[frame_idx]

                diff_from_baseline = calculate_color_difference(
                    current_frame_color_rgb,
                    baseline_pixel_color_rgb,
                    method=diff_method,
                )

                if diff_from_baseline > delta_e_deviation_threshold:
                    deviation_start_idx = frame_idx
                    first_color_of_event_rgb = current_frame_color_rgb
                    current_event_colors_rgb = [first_color_of_event_rgb]

                    temp_idx = frame_idx + 1
                    while temp_idx < num_frames:
                        next_frame_color_rgb = pixel_time_series[temp_idx]
                        diff_next_from_baseline = calculate_color_difference(
                            next_frame_color_rgb,
                            baseline_pixel_color_rgb,
                            method=diff_method,
                        )
                        diff_next_from_event_start = calculate_color_difference(
                            next_frame_color_rgb,
                            first_color_of_event_rgb,
                            method=diff_method,
                        )

                        if (
                            diff_next_from_baseline > delta_e_deviation_threshold
                            and diff_next_from_event_start
                            < delta_e_consistency_threshold
                        ):
                            current_event_colors_rgb.append(next_frame_color_rgb)
                            temp_idx += 1
                        else:
                            break

                    event_duration = temp_idx - deviation_start_idx

                    if (
                        min_leak_duration_frames
                        <= event_duration
                        <= max_leak_duration_frames
                    ):
                        avg_leak_event_color_rgb = np.mean(
                            current_event_colors_rgb, axis=0
                        ).astype(np.uint8)
                        collected_valid_leak_colors_rgb.append(avg_leak_event_color_rgb)

                    frame_idx = temp_idx
                else:
                    frame_idx += 1

            if collected_valid_leak_colors_rgb:
                final_pixel_color = get_mode_rgb_color(
                    np.array(collected_valid_leak_colors_rgb)
                )
                reconstructed_image[r, c] = final_pixel_color
            else:
                reconstructed_image[r, c] = (
                    baseline_pixel_color_rgb  # Fallback to median color
                )

    print("Background reconstruction finished.")
    return reconstructed_image


# --- Main execution block ---
if __name__ == "__main__":
    # !!! IMPORTANT: Replace these with your actual paths !!!
    video_file_path_str = "data/public/videos/24_i_livingroom_leaves_mp.mp4"
    mask_path_str = "data/public/masks/24_i_livingroom_mp.png"
    background_path_str = "data/public/backgrounds/livingroom.png"
    output_image_path_str = "img/reconstructed_background_median_transient.png"

    # --- Parameters for Transient Detection ---
    # These will likely need tuning based on your specific video's characteristics
    DEVIATION_THRESHOLD = 15.0  # For CIEDE2000. For RGB_SIMPLE, try 40-80+
    CONSISTENCY_THRESHOLD = 10.0  # For CIEDE2000. For RGB_SIMPLE, try 20-50+
    MIN_LEAK_DURATION = 1
    MAX_LEAK_DURATION = 6  # Key parameter to filter out longer events (e.g., person)
    USE_CIEDE2000_IF_AVAILABLE = True  # Set to False to force RGB diff for speed testing / if colour lib is problematic

    video_path = Path(video_file_path_str)
    mask_path = Path(mask_path_str)
    background_path = Path(background_path_str)
    output_path = Path(output_image_path_str)

    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
    elif "load_video" not in globals() or not callable(
        load_video
    ):  # Check if util.py was loaded
        print("Error: load_video function not available. Cannot proceed.")
    else:
        print(f"Loading video from: {video_path}")
        video_frames = load_video(video_path)

        if not video_frames:
            print("No frames loaded from video. Exiting.")
        else:
            print(f"Successfully loaded {len(video_frames)} frames.")

            reconstructed_bg = reconstruct_background_median_baseline_transient(
                video_frames,
                delta_e_deviation_threshold=DEVIATION_THRESHOLD,
                delta_e_consistency_threshold=CONSISTENCY_THRESHOLD,
                min_leak_duration_frames=MIN_LEAK_DURATION,
                max_leak_duration_frames=MAX_LEAK_DURATION,
                use_ciede2000=USE_CIEDE2000_IF_AVAILABLE,
            )

            print("\n--- Evaluating Reconstructed Background ---")
            if not (mask_path.exists() and background_path.exists()):
                print(
                    "Error: Mask or Background ground truth file not found for evaluation."
                )
                print(f"Mask path: {mask_path}")
                print(f"Background path: {background_path}")
            elif not all(
                callable(f)
                for f in [load_mask, load_background, metric_CIEDE2000, evaluate]
            ):
                print(
                    "Error: One or more evaluation functions from util.py are not available."
                )
            else:
                try:
                    gt_mask = load_mask(mask_path)
                    gt_background_image = load_background(background_path)

                    if reconstructed_bg.shape != gt_background_image.shape:
                        print(
                            f"Warning: Reconstructed shape {reconstructed_bg.shape} "
                            f"differs from GT background shape {gt_background_image.shape}."
                        )
                        # Attempt to resize reconstructed_bg to match gt_background_image for evaluation
                        # This might happen if video frames had a different resolution than the gt background.
                        # Or if the video loading/processing altered dimensions.
                        print(
                            f"Attempting to resize reconstructed image to {gt_background_image.shape[:2][::-1]} for evaluation."
                        )
                        reconstructed_bg_for_eval = cv2.resize(
                            reconstructed_bg,
                            (
                                gt_background_image.shape[1],
                                gt_background_image.shape[0],
                            ),
                        )
                    else:
                        reconstructed_bg_for_eval = reconstructed_bg

                    if gt_mask.shape[:2] != reconstructed_bg_for_eval.shape[:2]:
                        print(
                            f"Warning: Mask shape {gt_mask.shape[:2]} "
                            f"differs from reconstructed/GT background shape {reconstructed_bg_for_eval.shape[:2]}. Resizing mask."
                        )
                        gt_mask = cv2.resize(
                            gt_mask,
                            (
                                reconstructed_bg_for_eval.shape[1],
                                reconstructed_bg_for_eval.shape[0],
                            ),
                            interpolation=cv2.INTER_NEAREST,
                        )

                    delta_e_map = metric_CIEDE2000(
                        reconstructed_bg_for_eval, gt_background_image, gt_mask
                    )
                    evaluation_score = evaluate(delta_e_map, gt_mask)
                    print(
                        f"Evaluation score: {evaluation_score:.4f} (higher is better, 1.0 is perfect match in evaluated regions)"
                    )
                except Exception as e:
                    print(f"Error during evaluation: {e}")

            cv2.imshow(
                "Reconstructed Background (Median Baseline + Transient)",
                reconstructed_bg,
            )
            print(
                f"\nReconstructed background generated. Press any key to close the display window."
            )
            try:
                cv2.imwrite(str(output_path), reconstructed_bg)
                print(f"Reconstructed background saved to: {output_path}")
            except Exception as e:
                print(f"Error saving image: {e}")

            cv2.waitKey(0)
            cv2.destroyAllWindows()
