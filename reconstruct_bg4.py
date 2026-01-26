import numpy as np
import cv2
from pathlib import Path
import torch

# Assuming util.py is correctly set up
from util import load_video, metric_CIEDE2000, evaluate, load_mask, load_background


def reconstruct_background_gpu_uint8_chunked(
    video_numpy: np.ndarray,  # Expected shape: (frames, H, W, C), uint8
    deviation_threshold_rgb_sad: int,
    min_total_leak_frames: int,
    max_total_leak_frames: int,
    chunk_size_frames: int = 50,  # Number of frames to process in each chunk for SAD
    device_str: str = "cuda",
) -> np.ndarray:
    if not torch.cuda.is_available() and device_str == "cuda":
        print(
            "Warning: CUDA not available, falling back to CPU for PyTorch operations."
        )
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using PyTorch device: {device}")

    # 1. Video on GPU as uint8
    video_gpu_uint8 = torch.from_numpy(video_numpy).to(device)  # Already uint8
    num_frames, H, W, C = video_gpu_uint8.shape
    print(f"Video tensor on GPU: {video_gpu_uint8.shape}, {video_gpu_uint8.dtype}")

    # 2. Calculate Mode Frame (uint8 GPU)
    print("Calculating mode frame (baseline)...")
    mode_frame_gpu_uint8, _ = torch.mode(
        video_gpu_uint8, dim=0
    )  # Shape: (H, W, C), uint8
    # Promote mode_frame for subtractions within chunks
    mode_frame_gpu_int16 = mode_frame_gpu_uint8.to(torch.int16).unsqueeze(
        0
    )  # (1, H, W, C) for broadcasting
    print("Mode frame calculated.")

    # 3. SAD Calculation in Chunks & Build significant_deviation_mask_gpu
    print("Calculating SAD and significant deviations in chunks...")
    significant_deviation_mask_gpu = torch.empty(
        (num_frames, H, W), dtype=torch.bool, device=device
    )

    for i in range(0, num_frames, chunk_size_frames):
        frame_start = i
        frame_end = min(i + chunk_size_frames, num_frames)
        # print(f"  Processing SAD for frames {frame_start} to {frame_end-1}")

        current_video_chunk_uint8 = video_gpu_uint8[frame_start:frame_end]
        current_video_chunk_int16 = current_video_chunk_uint8.to(torch.int16)

        diff_int16 = (
            current_video_chunk_int16 - mode_frame_gpu_int16
        )  # Broadcasting mode_frame
        abs_diff_int16 = torch.abs(diff_int16)
        rgb_sad_chunk_int16 = abs_diff_int16.sum(
            dim=3
        )  # sum along channel dim -> (chunk_F, H, W)

        significant_deviation_mask_gpu[frame_start:frame_end] = (
            rgb_sad_chunk_int16 > deviation_threshold_rgb_sad
        )

        # Explicitly delete intermediate chunk tensors to free memory if tight
        del current_video_chunk_int16, diff_int16, abs_diff_int16, rgb_sad_chunk_int16
        if device_str == "cuda":
            torch.cuda.empty_cache()

    print(
        f"Found {significant_deviation_mask_gpu.sum()} significantly deviating pixel-frame instances."
    )

    # 4. Filter for Infrequency (GPU)
    num_deviation_frames_per_pixel_gpu = significant_deviation_mask_gpu.sum(
        dim=0
    )  # Shape: (H, W), type int/long

    valid_leak_pixel_mask_gpu = (
        num_deviation_frames_per_pixel_gpu >= min_total_leak_frames
    ) & (
        num_deviation_frames_per_pixel_gpu <= max_total_leak_frames
    )  # Bool, (H, W)
    print(
        f"Found {valid_leak_pixel_mask_gpu.sum()} pixels considered as potential leak areas."
    )

    # 5. Aggregate Leak Colors in Chunks
    print("Aggregating leak colors for valid pixels...")
    sum_of_leaked_colors_int32 = torch.zeros(
        (H, W, C), dtype=torch.int32, device=device
    )
    # Counts for averaging will be num_deviation_frames_per_pixel_gpu for relevant pixels

    for i in range(0, num_frames, chunk_size_frames):
        frame_start = i
        frame_end = min(i + chunk_size_frames, num_frames)
        # print(f"  Aggregating colors for frames {frame_start} to {frame_end-1}")

        current_video_chunk_uint8 = video_gpu_uint8[
            frame_start:frame_end
        ]  # (chunk_F, H, W, C)
        dev_mask_chunk = significant_deviation_mask_gpu[
            frame_start:frame_end
        ]  # (chunk_F, H, W)

        # Mask for this chunk, for pixels that are overall valid leak pixels
        #   and are deviating in this specific frame of the chunk
        active_leak_mask_chunk = dev_mask_chunk & valid_leak_pixel_mask_gpu.unsqueeze(
            0
        )  # (chunk_F, H, W)
        active_leak_mask_chunk_expanded = active_leak_mask_chunk.unsqueeze(
            -1
        ).expand_as(current_video_chunk_uint8)

        # Add colors from active leak pixel-frames to sum, promote video chunk to int32 for sum
        colors_to_sum_in_chunk = torch.where(
            active_leak_mask_chunk_expanded,
            current_video_chunk_uint8.to(torch.int32),
            torch.zeros(
                (1), dtype=torch.int32, device=device
            ),  # Add 0 where mask is false
        )
        sum_of_leaked_colors_int32 += colors_to_sum_in_chunk.sum(
            dim=0
        )  # Sum over frames in chunk -> (H,W,C)

        del (
            colors_to_sum_in_chunk,
            active_leak_mask_chunk,
            active_leak_mask_chunk_expanded,
        )
        if device_str == "cuda":
            torch.cuda.empty_cache()

    # Calculate average for the valid leak pixels
    # Denominator for average is num_deviation_frames_per_pixel_gpu (already int type)
    # We only care about pixels in valid_leak_pixel_mask_gpu

    # Prepare counts for division, ensuring no division by zero for non-target pixels
    counts_for_avg_int32 = num_deviation_frames_per_pixel_gpu.to(torch.int32)  # (H,W)
    # Ensure counts are at least 1 where valid_leak_pixel_mask_gpu is true, otherwise 1 to avoid div by zero
    # (though valid_leak_pixel_mask_gpu already implies count >= min_total_leak_frames)
    safe_counts_for_avg = torch.where(
        valid_leak_pixel_mask_gpu,
        counts_for_avg_int32,
        torch.ones_like(
            counts_for_avg_int32
        ),  # Use 1 for pixels we don't care about to prevent div by zero
    ).unsqueeze(
        -1
    )  # Expand to (H,W,1) for broadcasting

    avg_leaked_colors_float = (
        sum_of_leaked_colors_int32.float() / safe_counts_for_avg.float()
    )
    avg_leaked_colors_uint8 = avg_leaked_colors_float.clamp(0, 255).byte()

    # 6. Construct Final Image
    print("Constructing final background...")
    # Start with the mode frame, then update with aggregated leak colors where appropriate
    reconstructed_bg_gpu_uint8 = torch.where(
        valid_leak_pixel_mask_gpu.unsqueeze(-1).expand_as(
            mode_frame_gpu_uint8
        ),  # Condition (H,W,C)
        avg_leaked_colors_uint8,  # If True
        mode_frame_gpu_uint8,  # If False
    )

    # 7. Transfer result to CPU
    reconstructed_bg_numpy = reconstructed_bg_gpu_uint8.cpu().numpy()
    print("Reconstruction (uint8 chunked) finished.")
    return reconstructed_bg_numpy


# --- Main execution block ---
if __name__ == "__main__":
    video_file_path_str = "data/public/videos/24_i_livingroom_leaves_mp.mp4"
    mask_path_str = "data/public/masks/24_i_livingroom_mp.png"
    background_path_str = "data/public/backgrounds/livingroom.png"
    output_image_path_str = "img/reconstructed_bg_gpu_uint8_chunked.png"

    DEVIATION_THRESHOLD_RGB_SAD = 75  # For int16 SADs (range 0-765)
    MIN_TOTAL_LEAK_FRAMES = 2
    MAX_TOTAL_LEAK_FRAMES = 10
    CHUNK_SIZE_FRAMES = (
        64  # Adjust based on GPU memory. Smaller = slower but less peak mem per chunk.
    )
    PYTORCH_DEVICE = "cuda"

    video_path = Path(video_file_path_str)
    mask_path = Path(mask_path_str)
    background_path = Path(background_path_str)
    output_path = Path(output_image_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
    else:
        video_frames_list_cpu = load_video(video_path)
        if not video_frames_list_cpu:
            print(f"Failed to load video from {video_path}. Exiting.")
        else:
            video_numpy_cpu = np.array(video_frames_list_cpu)
            print(
                f"Video loaded. Shape: {video_numpy_cpu.shape}, Dtype: {video_numpy_cpu.dtype}"
            )

            if video_numpy_cpu.ndim != 4 or video_numpy_cpu.shape[3] != 3:
                print(
                    f"Error: Video NumPy array has unexpected shape: {video_numpy_cpu.shape}."
                )
            else:
                reconstructed_bg = reconstruct_background_gpu_uint8_chunked(
                    video_numpy_cpu,
                    deviation_threshold_rgb_sad=DEVIATION_THRESHOLD_RGB_SAD,
                    min_total_leak_frames=MIN_TOTAL_LEAK_FRAMES,
                    max_total_leak_frames=MAX_TOTAL_LEAK_FRAMES,
                    chunk_size_frames=CHUNK_SIZE_FRAMES,
                    device_str=PYTORCH_DEVICE,
                )

                print("\n--- Evaluating Reconstructed Background ---")
                try:
                    gt_mask = load_mask(mask_path)
                    gt_background_image = load_background(background_path)
                    reconstructed_bg_for_eval = reconstructed_bg
                    if reconstructed_bg.shape[:2] != gt_background_image.shape[:2]:
                        print(
                            f"Resizing reconstructed image from {reconstructed_bg.shape[:2]} to {gt_background_image.shape[:2]} for evaluation."
                        )
                        reconstructed_bg_for_eval = cv2.resize(
                            reconstructed_bg,
                            (
                                gt_background_image.shape[1],
                                gt_background_image.shape[0],
                            ),
                        )
                    if gt_mask.shape[:2] != reconstructed_bg_for_eval.shape[:2]:
                        print(
                            f"Resizing mask from {gt_mask.shape[:2]} to {reconstructed_bg_for_eval.shape[:2]} for evaluation."
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
                    print(f"Evaluation score: {evaluation_score:.4f}")
                except Exception as e:
                    print(f"Error during evaluation: {e}")

                cv2.imshow(
                    "Reconstructed Background (GPU uint8 Chunked)", reconstructed_bg
                )
                print(f"\nPress any key to close and save.")
                try:
                    cv2.imwrite(
                        str(output_path),
                        cv2.cvtColor(reconstructed_bg, cv2.COLOR_RGB2BGR),
                    )
                    print(f"Reconstructed background saved to: {output_path}")
                except Exception as e:
                    print(f"Error saving image: {e}")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
