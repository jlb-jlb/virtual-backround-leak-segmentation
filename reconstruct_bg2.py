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

from util import load_video


# --- Helper function to get mode color ---
def get_mode_rgb_color(pixel_time_series_data: np.ndarray) -> np.ndarray:
    """Calculates the mode (most frequent) color for a pixel's time series."""
    if pixel_time_series_data.ndim == 1:  # Should be (N, C)
        return pixel_time_series_data  # Or handle as error
    if pixel_time_series_data.shape[0] == 0:
        return np.array([0, 0, 0], dtype=np.uint8)  # Fallback for empty
    unique_colors, counts = np.unique(
        pixel_time_series_data, axis=0, return_counts=True
    )
    if len(unique_colors) > 0:
        return unique_colors[np.argmax(counts)].astype(np.uint8)
    return np.array([0, 0, 0], dtype=np.uint8)  # Fallback


# --- Color Difference Functions ---
def rgb_to_lab_for_colour_lib(rgb_color_uint8: np.ndarray) -> np.ndarray:
    """Converts a single RGB uint8 color to Lab float32 for colour-science."""
    # colour-science expects float32 scaled 0-1, assumed to be in sRGB space
    rgb_float = rgb_color_uint8.astype(np.float32) / 255.0

    # Step 1: Convert sRGB to XYZ
    # The sRGB_to_XYZ function handles the sRGB non-linearity.
    xyz_color = colour.sRGB_to_XYZ(rgb_float)

    # Step 2: Convert XYZ to Lab
    lab_color = colour.XYZ_to_Lab(xyz_color)

    return lab_color.astype(np.float32)  # Ensure it's float32 for delta_E


def calculate_color_difference(color1_rgb_uint8, color2_rgb_uint8, method="CIEDE2000"):
    if colour and method == "CIEDE2000":
        color1_lab = rgb_to_lab_for_colour_lib(color1_rgb_uint8)
        color2_lab = rgb_to_lab_for_colour_lib(color2_rgb_uint8)
        return colour.delta_E(color1_lab, color2_lab, method="CIE 2000")
    else:
        # Fallback to simple RGB difference (sum of absolute differences)
        # This is perceptually less accurate but avoids Lab conversion if 'colour' is not available
        return np.sum(
            np.abs(
                color1_rgb_uint8.astype(np.int16) - color2_rgb_uint8.astype(np.int16)
            )
        )


# --- Core reconstruction function ---
def reconstruct_background_via_transient_detection(
    video_frames_list: List[np.ndarray],
    delta_e_deviation_threshold: float = 15.0,  # How different from mode to be a deviation
    delta_e_consistency_threshold: float = 10.0,  # How consistent colors within a single leak event should be
    min_leak_duration_frames: int = 1,  # Min frames for a flicker to be valid
    max_leak_duration_frames: int = 7,  # Max frames for a flicker (key to ignore person)
    use_ciede2000: bool = True,
) -> np.ndarray:
    if not video_frames_list:
        raise ValueError("Input video_frames_list is empty.")
    num_frames = len(video_frames_list)
    if num_frames == 0:
        raise ValueError("Video has no frames.")

    first_frame = video_frames_list[0]
    height, width, channels = first_frame.shape
    video_data = np.array(video_frames_list)
    reconstructed_image = np.zeros((height, width, channels), dtype=np.uint8)

    diff_method = "CIEDE2000" if (colour and use_ciede2000) else "RGB_SIMPLE"
    print(
        f"Starting transient detection. Image: {width}x{height}, Frames: {num_frames}. Diff method: {diff_method}"
    )
    print(
        f"Params: DevThresh={delta_e_deviation_threshold}, ConsThresh={delta_e_consistency_threshold}, LeakDur=({min_leak_duration_frames}-{max_leak_duration_frames})"
    )

    for r in range(height):
        if r % max(1, height // 20) == 0 or r == height - 1:
            print(f"Processing row {r + 1}/{height}...")
        for c in range(width):
            pixel_time_series = video_data[:, r, c, :]
            mode_color_rgb = get_mode_rgb_color(pixel_time_series)

            collected_valid_leak_colors_rgb = []

            frame_idx = 0
            while frame_idx < num_frames:
                current_frame_color_rgb = pixel_time_series[frame_idx]

                # Is current color a deviation from mode?
                diff_from_mode = calculate_color_difference(
                    current_frame_color_rgb, mode_color_rgb, method=diff_method
                )

                if diff_from_mode > delta_e_deviation_threshold:
                    # Potential start of a leak event
                    deviation_start_idx = frame_idx
                    # The first color of this potential leak event
                    first_color_of_event_rgb = current_frame_color_rgb
                    current_event_colors_rgb = [first_color_of_event_rgb]

                    # Look ahead to see how long this deviation lasts
                    # and if its color remains consistent with the start of the event
                    temp_idx = frame_idx + 1
                    while temp_idx < num_frames:
                        next_frame_color_rgb = pixel_time_series[temp_idx]

                        diff_next_from_mode = calculate_color_difference(
                            next_frame_color_rgb, mode_color_rgb, method=diff_method
                        )
                        diff_next_from_event_start = calculate_color_difference(
                            next_frame_color_rgb,
                            first_color_of_event_rgb,
                            method=diff_method,
                        )

                        if (
                            diff_next_from_mode > delta_e_deviation_threshold
                            and diff_next_from_event_start
                            < delta_e_consistency_threshold
                        ):
                            # Still deviating from mode, and color is consistent with this event
                            current_event_colors_rgb.append(next_frame_color_rgb)
                            temp_idx += 1
                        else:
                            # Deviation ended (returned to mode-like or changed to a new, inconsistent color)
                            break

                    # temp_idx is now at the first frame *after* the consistent deviation sequence
                    event_duration = temp_idx - deviation_start_idx

                    if (
                        min_leak_duration_frames
                        <= event_duration
                        <= max_leak_duration_frames
                    ):
                        # This is a valid transient leak event
                        # Use the average color of this specific leak event
                        avg_leak_event_color_rgb = np.mean(
                            current_event_colors_rgb, axis=0
                        ).astype(np.uint8)
                        collected_valid_leak_colors_rgb.append(avg_leak_event_color_rgb)

                    frame_idx = temp_idx  # Advance main loop past this processed event
                else:
                    # Not a deviation from mode, move to the next frame
                    frame_idx += 1

            # After checking all frames for this pixel:
            if collected_valid_leak_colors_rgb:
                # If valid leaks were found, choose the mode of these collected leak colors
                # (in case multiple distinct leak events occurred for this pixel)
                final_pixel_color = get_mode_rgb_color(
                    np.array(collected_valid_leak_colors_rgb)
                )
                reconstructed_image[r, c] = final_pixel_color
            else:
                # No valid leaks found, use the original mode color (baseline)
                reconstructed_image[r, c] = mode_color_rgb

    print("Background reconstruction finished.")
    return reconstructed_image


# --- Main execution block ---
if __name__ == "__main__":
    video_file_path_str = "data/public/videos/24_i_livingroom_leaves_mp.mp4"  # !!! REPLACE WITH YOUR VIDEO PATH !!!
    output_image_path_str = "img/reconstructed_background_transient.png"

    # --- Parameters for Transient Detection ---
    # These will likely need tuning based on your specific video's characteristics (frame rate, leak appearance)
    # If using RGB_SIMPLE for difference, thresholds will be much larger (e.g., 30-100)
    # For CIEDE2000:
    DEVIATION_THRESHOLD = 15.0  # How different a pixel must be from its mode to be considered a deviation.
    CONSISTENCY_THRESHOLD = 10.0  # How similar colors within a single "flicker" event must be to each other.
    MIN_LEAK_DURATION = 1  # A flicker must last at least this many frames.
    MAX_LEAK_DURATION = 6  # A flicker must not last longer than this (to filter out people/longer changes).
    USE_CIEDE2000_IF_AVAILABLE = (
        True  # Set to False to force RGB diff for speed testing
    )

    video_path = Path(video_file_path_str)

    print(f"Loading video from: {video_file_path_str}")
    video_frames = load_video(Path(video_file_path_str))

    if not video_frames:
        print("No frames loaded. Exiting.")
    else:
        print(f"Successfully loaded {len(video_frames)} frames.")

        reconstructed_bg = reconstruct_background_via_transient_detection(
            video_frames,
            delta_e_deviation_threshold=DEVIATION_THRESHOLD,
            delta_e_consistency_threshold=CONSISTENCY_THRESHOLD,
            min_leak_duration_frames=MIN_LEAK_DURATION,
            max_leak_duration_frames=MAX_LEAK_DURATION,
            use_ciede2000=USE_CIEDE2000_IF_AVAILABLE,
        )
        from util import metric_CIEDE2000, evaluate, load_mask, load_background

        print("EVALUATIING RECONSTRUCTED BACKGROUND...")
        mask_path = Path("data/public/masks/24_i_livingroom_mp.png")
        background_path = Path("data/public/backgrounds/livingroom.png")
        mask_mask = load_mask(mask_path)
        background_image = load_background(background_path)

        metrix_matrix = metric_CIEDE2000(reconstructed_bg, background_image, mask_mask)
        evaluation_score = evaluate(metrix_matrix, mask_mask)
        print(
            f"Evaluation score: {evaluation_score:.4f} (higher is better, 1.0 is perfect match)"
        )

        cv2.imshow("Reconstructed Background (Transient Detection)", reconstructed_bg)
        print(f"\nReconstructed background generated. Press any key to close.")
        try:
            cv2.imwrite(output_image_path_str, reconstructed_bg)
            print(f"Reconstructed background saved to: {output_image_path_str}")
        except Exception as e:
            print(f"Error saving image: {e}")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
