import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import concurrent.futures
from justpfm import justpfm
from typing import Tuple, Dict
from scipy.optimize import curve_fit
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import img_as_ubyte
from skimage.util import view_as_windows


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
os.environ["OMP_NUM_THREADS"] = "1"

from src.utils.gaze_data import get_gaze_data
from src.utils.file import get_files_recursive, get_ids_from_file_path, get_set_str
from src.config import (
    SALIENCY_MAP_PFM_PATH,
    DEPTH_MAP_PFM_PATH,
    DEPTH_SEG_PFM_PATH,
    RAW_GAZE_FRAME_WIDTH,
    RAW_GAZE_FRAME_HEIGHT,
    GAZE_PROCESSED_PATH,
    MOTION_ABS_PFM_PATH,
    MOTION_OPF_PFM_PATH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    SETS_PATH,
)

DEFAULT_FRAME_STEP = 10
PATCH_SIZE = 10
N_NSEC_IN_SEC = 1e9
FPS = 25
SEED = 0


def get_fixation_data(
    frame_step: int,
) -> pd.DataFrame:
    """
    Get fixation data for videos.

    Returns:
        pd.DataFrame: Fixation data for videos
    """
    fixation_data_1 = get_gaze_data(
        experiment_ids=[1],
        set_ids=[0],
        fixation=True,
    )
    fixation_data_2 = get_gaze_data(
        experiment_ids=[2],
        fixation=True,
    )
    fixation_data = pd.concat([fixation_data_1, fixation_data_2])

    # Sort data and add Fixation Id
    fixation_data = fixation_data.sort_values(by=["SequenceId", "StartTimestamp_ns"])
    fixation_data["FixationId"] = np.arange(len(fixation_data))

    # Scale fixation coordinates
    fixation_data["X_px"] = fixation_data["X_px"] * IMAGE_WIDTH / RAW_GAZE_FRAME_WIDTH
    fixation_data["Y_px"] = fixation_data["Y_px"] * IMAGE_HEIGHT / RAW_GAZE_FRAME_HEIGHT

    # Get Frame Id per step
    fixation_data["FrameId"] = (
        fixation_data["TimeSinceStart_ns"] / N_NSEC_IN_SEC * FPS
    ).astype(int)
    fixation_data["FrameId"] = (fixation_data["FrameId"] / frame_step).astype(
        int
    ) * frame_step

    fixation_data = fixation_data[
        [
            "FixationId",
            "ExperimentId",
            "SetId",
            "SequenceId",
            "FrameId",
            "X_px",
            "Y_px",
        ]
    ]

    return fixation_data


def get_global_saliency_map() -> np.ndarray:
    """
    Get the global saliency map by merging all saliency maps

    Returns:
        np.ndarray: global saliency map
    """
    saliency_file_paths = Path(SALIENCY_MAP_PFM_PATH).rglob("*.pfm")
    saliency_file_paths = sorted(saliency_file_paths)

    global_saliency_map = None
    for saliency_file_path in tqdm(
        saliency_file_paths, desc="⌛ Loading saliency maps..."
    ):
        # Load map and merge
        saliency_map = justpfm.read_pfm(saliency_file_path)
        if global_saliency_map is None:
            global_saliency_map = saliency_map
        else:
            global_saliency_map += saliency_map

    # To unit distribution
    global_saliency_map /= global_saliency_map.sum()
    global_saliency_map = np.squeeze(global_saliency_map)

    return global_saliency_map


def gaussian_2d(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    sigma_x: float,
    sigma_y: float,
    A: float,
) -> np.ndarray:
    """
    A 2D Gaussian function

    Args:
        x (np.ndarray): x values
        y (np.ndarray): y values
        x0 (float): x center
        y0 (float): y center
        sigma_x (float): x standard deviation
        sigma_y (float): y standard deviation
        A (float): amplitude
    """

    return A * np.exp(
        -((x - x0) ** 2 / (2 * sigma_x**2) + (y - y0) ** 2 / (2 * sigma_y**2))
    )


def fit_gaussian(
    data: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Fit a 2D Gaussian to the given data

    Args:
        data (np.ndarray): 2D data to fit
        width (int): width of the data
        height (int): height of the data

    Returns:
        np.ndarray: fitted Gaussian
    """
    # Create a 2D grid
    x, y = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the arrays to fit with curve_fit
    x_data = np.vstack((x.ravel(), y.ravel()))
    z_data = data.ravel()

    # Start with an initial guess for (x0, y0, sigma_x, sigma_y, A)
    initial_guess = (width // 2, height // 2, 20, 30, 1)

    # Fit the Gaussian model to the data
    popt, _ = curve_fit(
        lambda xdata, x0, y0, sigma_x, sigma_y, A: gaussian_2d(
            xdata[0], xdata[1], x0, y0, sigma_x, sigma_y, A
        ),
        x_data,
        z_data,
        p0=initial_guess,
    )

    # Get the optimized parameters
    x0, y0, sigma_x, sigma_y, A = popt

    # Generate the fitted Gaussian using the optimized parameters
    fitted_gaussian = gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, A)

    return fitted_gaussian


def get_null_fixation_data(
    fixation_data: pd.DataFrame, fitted_gaussian: np.ndarray
) -> pd.DataFrame:
    """
    Get a null fixation data by sampling from a fitted Gaussian distribution

    Args:
        fixation_data (pd.Dataframe): The fixation data
        fitted_gaussian (np.ndarray): The fitted Gaussian distribution

    Returns:
        pd.DataFrame: The null fixation data
    """
    # Get the x and y coordinates
    x, y = np.meshgrid(
        np.arange(fitted_gaussian.shape[1]), np.arange(fitted_gaussian.shape[0])
    )

    # Flatten the arrays
    x = x.ravel()
    y = y.ravel()
    z = fitted_gaussian.ravel()

    # Normalize the z values
    z /= z.sum()

    # Sample from the fitted Gaussian distribution
    np.random.seed(SEED)
    sampled_indices = np.random.choice(np.arange(len(x)), size=len(fixation_data), p=z)
    sampled_x = x[sampled_indices]
    sampled_y = y[sampled_indices]

    # Create a new DataFrame
    null_fixation_data = fixation_data.copy()
    null_fixation_data["X_px"] = sampled_x
    null_fixation_data["Y_px"] = sampled_y

    return null_fixation_data


def get_base_and_null_fixation_data(
    frame_step: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get base and null fixation data

    Args:
        frame_step (int): Frame step for the fixation data

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Base and null fixation data
    """
    # Get fixation data
    fixation_data = get_fixation_data(frame_step)

    # Get null fixation data
    global_saliency_map = get_global_saliency_map()
    fitted_gaussian = fit_gaussian(
        data=global_saliency_map,
        width=global_saliency_map.shape[1],
        height=global_saliency_map.shape[0],
    )
    null_fixation_data = get_null_fixation_data(
        fixation_data=fixation_data, fitted_gaussian=fitted_gaussian
    )

    return fixation_data, null_fixation_data


def get_feature_data(
    folder_path: str,
    file_extension: str,
) -> Dict[int, Dict[int, Dict[int, np.ndarray]]]:
    file_paths = get_files_recursive(folder_path, f"*.{file_extension}")
    feature_data = {}
    for file_path in file_paths:
        # Get ids and create empty dict
        experiment_id, set_id, sequence_id = get_ids_from_file_path(file_path)
        if experiment_id not in feature_data:
            feature_data[experiment_id] = {}
        if set_id not in feature_data[experiment_id]:
            feature_data[experiment_id][set_id] = {}
        if sequence_id not in feature_data[experiment_id][set_id]:
            feature_data[experiment_id][set_id][sequence_id] = {}

        # Load data and add to dict
        data = justpfm.read_pfm(file_path)
        feature_data[experiment_id][set_id][sequence_id] = data

    return feature_data


def find_closest_frame(frame_id: int, motion_files: list) -> str:
    # Extract the frame numbers from the filenames and match the closest one
    frame_numbers = [
        int(f.stem.split('_')[0]) for f in motion_files
    ]  # assuming filenames are numbers, or can be extracted as such (N_x.pfm or N_y.pfm)
    closest_frame = min(frame_numbers, key=lambda x: abs(x - frame_id))

    # Find the corresponding file name for the closest frame
    closest_file = next(f for f in motion_files if int(f.stem.split('_')[0]) == closest_frame)
    return closest_file


def get_motion_feature_value(
    motion_abs_path: str,
    motion_opf_path: str,
    frame_id: int,
    row: pd.Series,
    r: int = PATCH_SIZE,
) -> Tuple[float, float]:
    motion_abs_files = sorted(Path(motion_abs_path).glob("*.pfm"))
    motion_opf_files_x = sorted(Path(motion_opf_path).glob("*_x.pfm"))
    motion_opf_files_y = sorted(Path(motion_opf_path).glob("*_y.pfm"))

    # Find closest frames for both motion abs and optical flow
    motion_abs_file = find_closest_frame(frame_id, motion_abs_files)
    motion_opf_file_x = find_closest_frame(frame_id, motion_opf_files_x)
    motion_opf_file_y = find_closest_frame(frame_id, motion_opf_files_y)

    # Load the motion images
    motion_abs_data = justpfm.read_pfm(motion_abs_file)
    motion_opf_data_x = justpfm.read_pfm(motion_opf_file_x)
    motion_opf_data_y = justpfm.read_pfm(motion_opf_file_y)

    # Extract the region of interest (ROI)
    x = row["X_px"]
    y = row["Y_px"]
    height, width = motion_abs_data.shape[:2]
    x_min = int(max(0, x - r))
    x_max = int(min(width, x + r + 1))
    y_min = int(max(0, y - r))
    y_max = int(min(height, y + r + 1))

    # Compute the average motion values within the patch
    motion_abs_value = motion_abs_data[y_min:y_max, x_min:x_max].mean()
    motion_opf_value_x = motion_opf_data_x[y_min:y_max, x_min:x_max].mean()
    motion_opf_value_y = motion_opf_data_y[y_min:y_max, x_min:x_max].mean()

    return motion_abs_value, motion_opf_value_x, motion_opf_value_y


def get_feature_value(
    data: Dict[int, Dict[int, Dict[int, np.ndarray]]] | np.ndarray,
    row: pd.Series,
    transform: callable = None,
    r: int = PATCH_SIZE,
) -> float:
    # Retrieve image
    if isinstance(data, Dict):
        experiment_id = row["ExperimentId"]
        set_id = row["SetId"]
        sequence_id = row["SequenceId"]
        data = data[experiment_id][set_id][sequence_id]

    # Get the shape and define the region of interest
    x = row["X_px"]
    y = row["Y_px"]
    height, width = data.shape[:2]
    x_min = int(max(0, x - r))
    x_max = int(min(width, x + r + 1))
    y_min = int(max(0, y - r))
    y_max = int(min(height, y + r + 1))

    # Apply transform if available
    if transform is not None:
        data = transform(data)

    # Extract the region of interest and calculate the mean
    roi = data[y_min:y_max, x_min:x_max]
    value = roi.mean()

    return value


def get_brightness(image: np.ndarray) -> float:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image


def get_colorfulness(image: np.ndarray) -> np.ndarray:
    rg = np.abs(image[:, :, 0] - image[:, :, 1])
    yb = np.abs(0.5 * (image[:, :, 0] + image[:, :, 1]) - image[:, :, 2])
    colorfulness_map = np.sqrt(rg**2 + yb**2)
    return colorfulness_map


def get_saturation(image: np.ndarray) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation_map = hsv_image[:, :, 1]
    return saturation_map


def get_hue(image: np.ndarray) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue_map = hsv_image[:, :, 0]
    return hue_map


def get_gradient_magnitude(image: np.ndarray) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude


def get_local_contrast(image: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((patch_size, patch_size)) / (patch_size**2)
    local_mean = cv2.filter2D(grayscale_image, -1, kernel)
    local_contrast = np.sqrt((grayscale_image - local_mean) ** 2)
    return local_contrast


def get_local_sharpness(image: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(grayscale_image, cv2.CV_64F)
    kernel = np.ones((patch_size, patch_size)) / (patch_size**2)
    local_sharpness = cv2.filter2D(np.abs(laplacian), -1, kernel)
    return local_sharpness


def get_contrast_sensitivity(
    image: np.ndarray, patch_size: int = PATCH_SIZE
) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((patch_size, patch_size)) / (patch_size**2)
    local_min = cv2.erode(grayscale_image, kernel, iterations=1)
    local_max = cv2.dilate(grayscale_image, kernel, iterations=1)
    contrast_sensitivity = (local_max - local_min) / (local_max + local_min + 1e-5)
    return contrast_sensitivity


def get_local_variance(image: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((patch_size, patch_size)) / (patch_size**2)
    local_mean = cv2.filter2D(grayscale_image, -1, kernel)
    local_squared_mean = cv2.filter2D(grayscale_image**2, -1, kernel)
    local_variance = local_squared_mean - local_mean**2
    return local_variance


def get_local_energy(image: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
    local_energy = np.sqrt(gradient_x**2 + gradient_y**2)
    kernel = np.ones((patch_size, patch_size)) / (patch_size**2)
    local_energy = cv2.filter2D(local_energy, -1, kernel)
    return local_energy


def get_edge_density(image: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(grayscale_image, 100, 200)
    kernel = np.ones((patch_size, patch_size)) / (patch_size**2)
    edge_density = cv2.filter2D(edges.astype(np.float32), -1, kernel)
    return edge_density


def process_frame(
    group,
    video_path,
    motion_abs_path,
    motion_opf_path,
):
    # Get the frame data
    video = cv2.VideoCapture(video_path)
    frame_id = group["FrameId"].values[0]
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = video.read()
    if not ret:
        raise ValueError(f"❌ Frame {frame_id} could not be read.")

    # Add motion features
    group["MotionAbs"], group["MotionOpf_x"], group["MotionOpf_y"] = zip(
        *group.apply(
            lambda x: get_motion_feature_value(
                motion_abs_path=motion_abs_path,
                motion_opf_path=motion_opf_path,
                frame_id=frame_id,
                row=x,
            ),
            axis=1,
        )
    )

    # Add image features
    group["Brightness"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_brightness), axis=1
    )
    group["Colorfulness"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_colorfulness),
        axis=1,
    )
    group["Saturation"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_saturation), axis=1
    )
    group["Hue"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_hue), axis=1
    )
    group["GradientMagnitude"] = group.apply(
        lambda x: get_feature_value(
            data=frame, row=x, transform=get_gradient_magnitude
        ),
        axis=1,
    )
    group["LocalContrast"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_local_contrast),
        axis=1,
    )
    group["LocalSharpness"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_local_sharpness),
        axis=1,
    )
    group["ContrastSensitivity"] = group.apply(
        lambda x: get_feature_value(
            data=frame, row=x, transform=get_contrast_sensitivity
        ),
        axis=1,
    )
    group["LocalVariance"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_local_variance),
        axis=1,
    )
    group["LocalEnergy"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_local_energy),
        axis=1,
    )
    group["EdgeDensity"] = group.apply(
        lambda x: get_feature_value(data=frame, row=x, transform=get_edge_density),
        axis=1,
    )

    video.release()

    return group


def process_video(group):
    # Get group ids and video
    experiment_id = group["ExperimentId"].values[0]
    set_id = group["SetId"].values[0]
    sequence_id = group["SequenceId"].values[0]
    set_str = get_set_str(experiment_id=experiment_id, set_id=set_id)
    video_path = (
        f"{SETS_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}.mp4"
    )
    motion_abs_path = f"{MOTION_ABS_PFM_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}"
    motion_opf_path = f"{MOTION_OPF_PFM_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}"

    # Process frames
    grouped_data = group.groupby("FrameId")
    frame_rows = [row for _, row in grouped_data]
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        with tqdm(
            total=len(frame_rows),
            desc=f"⌛ Processing image features of experiment {experiment_id}, set {set_id}, sequence {sequence_id}...",
        ) as bar:
            futures = [
                executor.submit(
                    process_frame,
                    row,
                    video_path,
                    motion_abs_path=motion_abs_path,
                    motion_opf_path=motion_opf_path,
                )
                for row in frame_rows
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                bar.update(1)

    return pd.concat(results)


def get_fixation_features(data: pd.DataFrame) -> pd.DataFrame:
    # Get depth features
    bar = tqdm(total=2, desc="⌛ Loading depth features...")
    depth_maps = get_feature_data(DEPTH_MAP_PFM_PATH, "pfm")
    bar.update(1)
    depth_segs = get_feature_data(DEPTH_SEG_PFM_PATH, "pfm")
    bar.update(1)
    bar.close()

    # Add depth features
    tqdm.pandas(desc="⌛ Processing depth feature...")
    data["Depth"] = data.progress_apply(
        lambda x: get_feature_value(data=depth_maps, row=x),
        axis=1,
    )
    tqdm.pandas(desc="⌛ Processing depth segmentation feature...")
    data["DepthSeg"] = data.progress_apply(
        lambda x: get_feature_value(data=depth_segs, row=x),
        axis=1,
    )

    # Group by experiment, set, and sequence
    grouped_data = data.groupby(["ExperimentId", "SetId", "SequenceId"])
    results = grouped_data.apply(lambda group: process_video(group))
    results = results.reset_index(drop=True)

    return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get global saliency maps.")
    parser.add_argument(
        "--frame-step",
        "-fs",
        type=int,
        default=DEFAULT_FRAME_STEP,
        help="Frame step for the fixation data.",
    )

    return parser.parse_args()


def main() -> None:
    # Parse command line arguments
    args = parse_arguments()
    frame_step = args.frame_step

    # Get data and their features
    fixation_data, null_fixation_data = get_base_and_null_fixation_data(frame_step)
    print(
        f"✅ Base and null fixation data generated with {len(fixation_data):,} fixations each."
    )

    fixation_data = get_fixation_features(fixation_data)
    null_fixation_data = get_fixation_features(null_fixation_data)
    print(f"✅ Features extracted.")

    # Write data to disk
    fixation_file_path = f"{GAZE_PROCESSED_PATH}/features_fixation_data.csv"
    null_fixation_file_path = f"{GAZE_PROCESSED_PATH}/features_null_fixation_data.csv"
    fixation_data.to_csv(fixation_file_path, index=False)
    null_fixation_data.to_csv(null_fixation_file_path, index=False)
    print(f"✅ Fixation data saved to {fixation_file_path}")


if __name__ == "__main__":
    main()
