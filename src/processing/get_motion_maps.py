import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import torch
import argparse
import warnings
import numpy as np
from torch import nn
from tqdm import tqdm
from justpfm import justpfm
from multiprocessing import Pool
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

from src.utils.file import get_files_recursive, get_ids_from_file_path, get_set_str
from src.config import (
    SETS_PATH,
    MOTION_ABS_IMG_PATH,
    MOTION_ABS_PFM_PATH,
    MOTION_OPF_IMG_PATH,
    MOTION_OPF_PFM_PATH,
    DEVICE,
)

DEFAULT_FRAME_STEP = 24
DEFAULT_H = 10
RAFT_SHAPE_RATIO = 2
RAFT_MULTIPLE = 8
DEFAULT_TEMPLATE_WINDOW_SIZE = 7
DEFAULT_SEARCH_WINDOW_SIZE = 21
DEFAULT_KERNEL_SIZE = 7


def get_optical_flow_model() -> nn.Module:
    """
    Get the optical flow model.

    Returns:
        nn.Module: The optical flow model
    """

    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights).to(DEVICE)
    model.eval()
    print(f"✅ Loaded RAFT model with weights {weights}.")

    return model


def estimate_raft_flow(
    raft_model: nn.Module,
    first_frame: np.ndarray,
    last_frame: np.ndarray,
    shape_ratio: int = RAFT_SHAPE_RATIO,
    multiple: int = RAFT_MULTIPLE,
) -> np.ndarray:
    """
    Estimate optical flow between two frames using RAFT.

    Args:
        raft_model: Preloaded RAFT model.
        first_frame: The first frame in the sequence step.
        last_frame: The last frame in the sequence step.

    Returns:
        np.ndarray: The estimated optical flow.
    """
    # Get tensor from frame
    old_shape = (first_frame.shape[0], first_frame.shape[1])
    new_shape = (
        (old_shape[0] // (shape_ratio * multiple)) * multiple,
        (old_shape[1] // (shape_ratio * multiple)) * multiple,
    )
    first_tensor = (
        torch.from_numpy(first_frame).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    )
    last_tensor = (
        torch.from_numpy(last_frame).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    )
    transforms = T.Compose(
        [
            T.Resize(size=(new_shape[0], new_shape[1])),
            T.Lambda(lambda x: x / 255.0),  # [0, 255] to [0, 1]
            T.Normalize(mean=[0.5], std=[0.5]),  # [0, 1] to [-1, 1]
        ]
    )
    first_tensor = transforms(first_tensor)
    last_tensor = transforms(last_tensor)

    with torch.no_grad():
        optical_flow = raft_model(first_tensor, last_tensor)
        optical_flow = optical_flow[0].squeeze().cpu().numpy()

        # Resize flow to original shape
        optical_flow = optical_flow.transpose(1, 2, 0)
        optical_flow = cv2.resize(
            optical_flow, (old_shape[1], old_shape[0]), interpolation=cv2.INTER_LINEAR
        )

    # Free GPU memory
    del first_tensor
    del last_tensor
    torch.cuda.empty_cache()

    return optical_flow


def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Applies the required transformations (grayscale, denoising, and blurring) to a frame.

    Args:
        frame (np.ndarray): The frame to process.

    Returns:
        np.ndarray: The processed frame.
    """

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.fastNlMeansDenoising(
        frame, None, DEFAULT_H, DEFAULT_TEMPLATE_WINDOW_SIZE, DEFAULT_SEARCH_WINDOW_SIZE
    )
    frame = cv2.GaussianBlur(frame, (DEFAULT_KERNEL_SIZE, DEFAULT_KERNEL_SIZE), 0)

    return frame


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow to a color-coded image using the HSV color space.

    Args:
        flow (np.ndarray): Optical flow array with shape (H, W, 2), where the last dimension represents (dx, dy).

    Returns:
        np.ndarray: The color-coded image representing flow direction and magnitude in RGB format.
    """
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = (
        angle * 180 / np.pi / 2
    )  # Angle in degrees, scaled to [0, 180] for hue
    hsv[..., 1] = 1
    hsv[..., 2] = magnitude
    rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return (rgb_flow * 255).astype(np.uint8)


def get_motion_maps(
    video_file_path: str,
    frame_step: int,
    raft_model: nn.Module,
    bar: tqdm,
) -> None:
    # Get output folders
    experiment_id, session_id, sequence_id = get_ids_from_file_path(video_file_path)
    set_str = get_set_str(experiment_id, session_id)
    output_abs_pfm_folder_path = f"{MOTION_ABS_PFM_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}"
    os.makedirs(output_abs_pfm_folder_path, exist_ok=True)
    output_abs_img_folder_path = f"{MOTION_ABS_IMG_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}"
    os.makedirs(output_abs_img_folder_path, exist_ok=True)
    output_opf_pfm_folder_path = f"{MOTION_OPF_PFM_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}"
    os.makedirs(output_opf_pfm_folder_path, exist_ok=True)
    output_opf_img_folder_path = f"{MOTION_OPF_IMG_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}"
    os.makedirs(output_opf_img_folder_path, exist_ok=True)

    # Iterate over frames
    video = cv2.VideoCapture(video_file_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    pool = Pool(processes=os.cpu_count())
    accumulated_rgb_frames = []

    # Check if the video has already been processed, and skip if so
    output_abs_pfm_file_paths = get_files_recursive(output_abs_pfm_folder_path, "*.pfm")
    outout_abs_img_file_paths = get_files_recursive(output_abs_img_folder_path, "*.png")
    output_opf_pfm_file_paths = get_files_recursive(output_opf_pfm_folder_path, "*.pfm")
    output_opf_img_file_paths = get_files_recursive(output_opf_img_folder_path, "*.png")
    if (
        len(output_abs_pfm_file_paths) == total_frames // frame_step
        and len(outout_abs_img_file_paths) == total_frames // frame_step
        and len(output_opf_pfm_file_paths) == total_frames // frame_step * 2 # x and y
        and len(output_opf_img_file_paths) == total_frames // frame_step
    ):
        bar.set_postfix(
            {
                "skipped video": f"{Path(video_file_path).name}",
            }
        )
        return
    
    frame_id = -1
    while True:
        frame_id += 1

        # Read frames
        ret, rgb_frame = video.read()
        if not ret:
            break

        accumulated_rgb_frames.append(rgb_frame)

        if frame_id % frame_step == 0 and len(accumulated_rgb_frames) > 1:
            # Update accumulated frames
            curr_accumulated_rgb_frames = accumulated_rgb_frames.copy()
            accumulated_rgb_frames = []

            # Check if the files have already been processed, and skip if so
            output_pfm_file_path = f"{output_abs_pfm_folder_path}/{frame_id}.pfm"
            output_image_file_path = f"{output_abs_img_folder_path}/{frame_id}.png"
            if os.path.exists(output_pfm_file_path) and os.path.exists(output_image_file_path):
                bar.set_postfix(
                    {
                        "skipped frame": f"{frame_id}/{total_frames}",
                    }
                )
                continue

            # Compute absolute difference map
            accumulated_processed_frames = pool.map(
                process_frame, curr_accumulated_rgb_frames
            )
            absolute_difference_map = np.mean(
                np.abs(np.diff(accumulated_processed_frames, axis=0)), axis=0
            ).astype(np.float32)

            # Save absolute difference map
            justpfm.write_pfm(
                file_name=output_pfm_file_path, data=absolute_difference_map
            )

            absolute_difference_map = cv2.normalize(
                absolute_difference_map, None, 0, 255, cv2.NORM_MINMAX
            )
            cv2.imwrite(output_image_file_path, absolute_difference_map)

            # Compute optical flow
            first_frame = curr_accumulated_rgb_frames[0]
            last_frame = curr_accumulated_rgb_frames[-1]
            optical_flow = estimate_raft_flow(
                raft_model=raft_model, first_frame=first_frame, last_frame=last_frame
            )
            optical_flow_x = optical_flow[:, :, 0]
            optical_flow_y = optical_flow[:, :, 1]

            # Save optical flow
            output_pfm_file_path_x = f"{output_opf_pfm_folder_path}/{frame_id}_x.pfm"
            output_pfm_file_path_y = f"{output_opf_pfm_folder_path}/{frame_id}_y.pfm"
            justpfm.write_pfm(file_name=output_pfm_file_path_x, data=optical_flow_x)
            justpfm.write_pfm(file_name=output_pfm_file_path_y, data=optical_flow_y)

            output_image_file_path = f"{output_opf_img_folder_path}/{frame_id}.png"
            optical_flow_color = flow_to_color(optical_flow)
            cv2.imwrite(output_image_file_path, optical_flow_color)
            
        bar.set_postfix(
            {
                "processed frame": f"{frame_id}/{total_frames}",
            }
        )

    video.release()
    cv2.destroyAllWindows()


def get_raft_maps():
    pass


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get motion maps from videos.")

    parser.add_argument(
        "--frame-step",
        "-s",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="The step size between frames.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for estimating motion maps from videos.

    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    # Parse arguments
    args = parse_arguments()
    frame_step = args.frame_step

    # Get optical flow model
    raft_model = get_optical_flow_model()

    # Get video file paths and compute motion maps
    video_file_paths = get_files_recursive(SETS_PATH, "*.mp4")
    bar = tqdm(video_file_paths, desc="⌛ Computing motion maps...", unit="video")
    for video_file_path in bar:
        get_motion_maps(
            video_file_path=video_file_path,
            frame_step=frame_step,
            raft_model=raft_model,
            bar=bar,
        )


if __name__ == "__main__":
    main()
