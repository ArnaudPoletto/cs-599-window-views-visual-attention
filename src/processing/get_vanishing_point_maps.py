import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from typing import List
from justpfm import justpfm

from src.utils.kde import get_kde_density
from src.utils.file import get_files_recursive, get_ids_from_file_path, get_session_str
from src.config import (
    IMAGES_PATH,
    VANISHING_POINT_MAP_IMG_PATH,
    VANISHING_POINT_MAP_PFM_PATH,
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
)

KERNEL_SIZE = 7
CANNY_THRESHOLD1 = 100
CANNY_THRESHOLD2 = 150
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 80
MIN_LINE_LENGTH = 100
MAX_LINE_GAP = 10
D_ALPHA_DEG = 5
PROXIMITY_THRESHOLD = 25
MIN_LEADING_LINES = 10

KDE_WIDTH = 192
KDE_HEIGHT = 96
KDE_BANDWIDTH = 10
N_SAMPLES = -1


def are_lines_close(line1: np.ndarray, line2: np.ndarray, threshold: int) -> bool:
    """
    Check if two lines are close to each other.

    Args:
        line1 (np.ndarray): The first line.
        line2 (np.ndarray): The second line.
        threshold (int): The proximity threshold.
    """
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]

    # Check the Euclidean distance between endpoints of the two lines
    dist1 = np.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
    dist2 = np.sqrt((x2 - x4) ** 2 + (y2 - y4) ** 2)

    return dist1 < threshold and dist2 < threshold


def get_leading_lines(
    image: np.array,
    kernel_size: int = KERNEL_SIZE,
    canny_threshold1: int = CANNY_THRESHOLD1,
    canny_threshold2: int = CANNY_THRESHOLD2,
    hough_rho: int = HOUGH_RHO,
    hough_theta: float = HOUGH_THETA,
    hough_threshold: int = HOUGH_THRESHOLD,
    min_line_length: int = MIN_LINE_LENGTH,
    max_line_gap: int = MAX_LINE_GAP,
    d_alpha_deg: int = D_ALPHA_DEG,
    proximity_threshold: int = PROXIMITY_THRESHOLD,
) -> List[np.ndarray]:
    """
    Get the leading lines from a depth map.

    Args:
        image (np.array): The depth map.
        kernel_size (int, optional): The kernel size for the Gaussian filter. Defaults to KERNEL_SIZE.
        canny_threshold1 (int, optional): The first threshold for the Canny edge detection. Defaults to CANNY_THRESHOLD1.
        canny_threshold2 (int, optional): The second threshold for the Canny edge detection. Defaults to CANNY_THRESHOLD2.
        min_line_length (int, optional): The minimum line length. Defaults to MIN_LINE_LENGTH.
        max_line_gap (int, optional): The maximum line gap. Defaults to MAX_LINE_GAP.
        d_alpha_deg (int, optional): The angle threshold for removing vertical and horizontal lines, in degrees. Defaults to D_ALPHA_DEG.
        proximity_threshold (int, optional): The proximity threshold for removing redundant lines. Defaults to PROXIMITY_THRESHOLD.

    """
    # Normalize the depth map for edge detection
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Smooth the image using a Gaussian filter and apply Canny edge detection
    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    edges = cv2.Canny(image, threshold1=canny_threshold1, threshold2=canny_threshold2)

    # Apply the probabilistic Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(
        edges,
        rho=hough_rho,
        theta=hough_theta,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    if lines is None:
        return []

    # Remove vertical and horizontal lines
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if 90 - d_alpha_deg < angle < 90 + d_alpha_deg:
            continue
        if -d_alpha_deg < angle < d_alpha_deg:
            continue
        filtered_lines.append(line)
    lines = filtered_lines

    # Remove redundant lines
    filtered_lines = []
    for i, line in enumerate(lines):
        is_redundant = False
        for j in range(i):
            if are_lines_close(line, lines[j], proximity_threshold):
                is_redundant = True
                break
        if not is_redundant:
            filtered_lines.append(line)
    lines = filtered_lines

    return lines


def line_intersection(line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
    """
    Compute the intersection point between two lines.

    Args:
        line1 (np.ndarray): The first line.
        line2 (np.ndarray): The second line.

    Returns:
        np.ndarray: The intersection point.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Compute determinant
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1
    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3
    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None

    # Solve for intersection point
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant

    return int(x), int(y)


def get_vanishing_point_map(
    leading_lines: List[np.ndarray],
    image_width: int,
    image_height: int,
    kde_width: int,
    kde_height: int,
    kde_bandwidth: float,
    n_samples: int,
) -> np.array:
    """
    Get the vanishing point map from the leading lines.

    Args:
        leading_lines (List[np.ndarray]): The leading lines.
        image_width (int): The width of the image.
        image_height (int): The height of the image.
        kde_width (int): The width of the KDE map.
        kde_height (int): The height of the KDE map.
        kde_bandwidth (float): The bandwidth of the KDE.
        n_samples (int): Number of samples to use.

    Returns:
        np.array: The vanishing point map.
    """
    intersection_points = []

    # Find intersections between all pairs of lines
    for i in range(len(leading_lines)):
        for j in range(i + 1, len(leading_lines)):
            # Get intersection point
            line1 = leading_lines[i][0]
            line2 = leading_lines[j][0]
            intersection = line_intersection(line1, line2)
            if not intersection:
                continue

            # Only consider intersections within the image bounds
            x, y = intersection
            if 0 <= x < image_width and 0 <= y < image_height:
                intersection_points.append(intersection)

    # Return a blank map if not enough intersection points are found
    if len(intersection_points) <= MIN_LEADING_LINES:
        return np.zeros((image_height, image_width))

    # Scale intersection points to fit the KDE dimensions
    intersection_points = np.array(intersection_points)
    width_ratio = kde_width / image_width
    height_ratio = kde_height / image_height
    intersection_points[:, 0] = (intersection_points[:, 0] * width_ratio).astype(int)
    intersection_points[:, 1] = (intersection_points[:, 1] * height_ratio).astype(int)

    # Apply kernel density estimation on the intersection points
    kde_density = get_kde_density(
        coordinates=intersection_points,
        kde_width=kde_width,
        kde_height=kde_height,
        kde_bandwidth=kde_bandwidth,
        apply_exponential=True,
        n_samples=n_samples,
    )

    # Resize the KDE values to the size of the depth map
    kde_density = cv2.resize(kde_density, (image_width, image_height))

    return kde_density


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get vanishing point maps.")
    parser.add_argument(
        "--kde-width",
        "-kw",
        type=int,
        default=KDE_WIDTH,
        help="Width of the KDE map.",
    )
    parser.add_argument(
        "--kde-height",
        "-kh",
        type=int,
        default=KDE_HEIGHT,
        help="Height of the KDE map.",
    )
    parser.add_argument(
        "--kde-bandwidth",
        "-kb",
        type=float,
        default=KDE_BANDWIDTH,
        help="Bandwidth of the kernel density estimation.",
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=N_SAMPLES,
        help="Number of samples to use for the KDE.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for computing the vanishing point maps for each sequence.
    """
    args = parse_arguments()
    kde_width = args.kde_width
    kde_height = args.kde_height
    kde_bandwidth = args.kde_bandwidth
    n_samples = args.n_samples

    image_file_paths = get_files_recursive(IMAGES_PATH, "*.png")
    for image_file_path in tqdm(
        image_file_paths, desc="⌛ Computing vanishing point maps...", unit="image"
    ):
        image = cv2.imread(image_file_path)

        leading_lines = get_leading_lines(image)
        vanishing_point_map = get_vanishing_point_map(
            leading_lines=leading_lines,
            image_width=IMAGE_WIDTH,
            image_height=IMAGE_HEIGHT,
            kde_width=kde_width,
            kde_height=kde_height,
            kde_bandwidth=kde_bandwidth,
            n_samples=n_samples,
        )
        vanishing_point_map = vanishing_point_map.astype(np.float32)

        # Save vanishing point map
        experiment_id, session_id, sequence_id = get_ids_from_file_path(image_file_path)
        session_str = get_session_str(experiment_id, session_id)
        vanishing_point_map_pfm_path = f"{VANISHING_POINT_MAP_PFM_PATH}/experiment{experiment_id}/{session_str}/scene{sequence_id}.pfm"
        os.makedirs(os.path.dirname(vanishing_point_map_pfm_path), exist_ok=True)
        justpfm.write_pfm(
            file_name=vanishing_point_map_pfm_path, data=vanishing_point_map
        )

        vanishing_point_map_img_path = f"{VANISHING_POINT_MAP_IMG_PATH}/experiment{experiment_id}/{session_str}/scene{sequence_id}.png"
        vanishing_point_map = cv2.normalize(
            vanishing_point_map, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        os.makedirs(os.path.dirname(vanishing_point_map_img_path), exist_ok=True)
        cv2.imwrite(vanishing_point_map_img_path, vanishing_point_map)

    print(f"✅ Vanishing point maps computed and saved at {Path(VANISHING_POINT_MAP_PFM_PATH).resolve()}.") 


if __name__ == "__main__":
    main()
