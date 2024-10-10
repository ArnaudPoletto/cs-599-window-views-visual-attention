import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from skimage import graph
from skimage import filters
from justpfm import justpfm
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from skimage.segmentation import slic

from src.utils.file import get_files_recursive
from src.config import (
    DEPTH_MAP_PFM_PATH,
)


DEFAULT_N_CLUSTERS = 5
KERNEL_SIZE = 5
N_SEGMENTS = 1000
DEPTH_WEIGHT_POWER = 4.0
FOREGROUND_SCORE_THRESHOLD = 0.2
SIMILARITY_THRESHOLD = 2.0
SKY_THRESHOLD = 0.0


def get_foreground_mask(
    depth_map: np.ndarray,
    n_segments: int = N_SEGMENTS,
    depth_weight_power: float = DEPTH_WEIGHT_POWER,
    foreground_score_threshold: float = FOREGROUND_SCORE_THRESHOLD,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    eps: float = 1e-2,
) -> np.ndarray:
    # Get superpixels and their means
    superpixels = slic(
        depth_map, n_segments=n_segments, compactness=0.001, start_label=1
    )
    superpixel_means = np.array(
        [np.mean(depth_map[superpixels == i]) for i in np.unique(superpixels)]
    )

    # Construct a graph of neighboring pixels
    rag = graph.rag_mean_color(depth_map, superpixels)

    # Identify initial foreground superpixels based on the ratio threshold
    initial_foreground = set()
    max_depth = np.max(superpixel_means)
    foreground_mask = np.zeros_like(depth_map, dtype=bool)
    for region in rag.nodes:
        current_depth = superpixel_means[region - 1]
        depth_weight = current_depth / max_depth
        for neighbor in rag.neighbors(region):
            neighbor_depth = superpixel_means[neighbor - 1]

            # Get foreground superpixels based on the ratio threshold
            foreground_score = abs(current_depth - neighbor_depth) / (
                current_depth + neighbor_depth + eps
            )
            foreground_score *= depth_weight**depth_weight_power
            if foreground_score > foreground_score_threshold:
                foreground_mask[superpixels == region] = True
                initial_foreground.add(region)
                break

    # Propagate the foreground label to interior superpixels that are similar in depth
    # Visited nodes inherit from the origin node depth value, so that the propagation is consistent
    visited = set(initial_foreground)
    queue = [(region, superpixel_means[region - 1]) for region in initial_foreground]
    while queue:
        region, origin_depth = queue.pop(0)

        # Iterate through neighbors of the current region
        for neighbor in rag.neighbors(region):
            # Skip already visited regions
            if neighbor in visited:
                continue

            # Propagate the foreground label if the neighbor is closer than the origin
            # or if the neighbor is similar to the origin in depth
            neighbor_depth = superpixel_means[neighbor - 1]
            similarity = abs(origin_depth - neighbor_depth)
            if neighbor_depth > origin_depth or similarity < similarity_threshold:
                foreground_mask[superpixels == neighbor] = True
                visited.add(neighbor)
                queue.append((neighbor, origin_depth))

    return foreground_mask


def get_sky_mask(
    depth_map: np.ndarray,
    sky_threshold: float = SKY_THRESHOLD,
) -> np.ndarray:
    sky_mask = depth_map <= sky_threshold

    return sky_mask


def segment_depth(
    file_path: str,
    n_clusters: int,
    kernel_size: int = KERNEL_SIZE,
) -> None:
    """
    Segment the given pfm depth image into clusters. Optionally, assign the foreground cluster a value of 0, while the other n_clusters are labeled from 1 (near) to n_clusters (far).

    Args:
        file_path (str): Path to the pfm depth image file.
        n_clusters (int): Number of clusters to segment the depth images into, excluding the foreground cluster.
        kernel_size (int): Size of the gaussian kernel to apply to the depth map.

    Raises:
        ValueError: If no background values are found in the depth map.
    """
    # Load pfm file
    depth_map = justpfm.read_pfm(file_path)

    # Apply gaussian blur to depth map
    depth_map = cv2.GaussianBlur(depth_map, (kernel_size, kernel_size), 0)[
        ..., np.newaxis
    ]

    # Get foreground and sky masks
    foreground_mask = get_foreground_mask(depth_map)
    sky_mask = get_sky_mask(depth_map)

    # Find background values to be clustered
    background_values = depth_map[~foreground_mask & ~sky_mask].flatten().reshape(-1, 1)
    if background_values.size == 0:
        raise ValueError(f"❌ No background values found in {file_path}.")

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(background_values)

    # Get sorted cluster values
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    cluster_values = np.zeros_like(kmeans.labels_)
    for i, idx in enumerate(sorted_indices):
        cluster_values[kmeans.labels_ == idx] = i + 1

    # Get segmented map
    segmented_map = np.zeros_like(depth_map)
    segmented_map[~foreground_mask & ~sky_mask] = cluster_values

    # Assign sky values to 0 and foreground values to n_clusters + 1
    segmented_map[sky_mask] = 0
    segmented_map[foreground_mask] = n_clusters + 1

    # Save segmented depth map as pfm and image files
    segmented_pfm_file_path = file_path.replace("map_pfm", "seg_pfm")
    os.makedirs(os.path.dirname(segmented_pfm_file_path), exist_ok=True)
    justpfm.write_pfm(file_name=segmented_pfm_file_path, data=segmented_map)

    segmented_img_file_path = file_path.replace("map_pfm", "seg_img").replace(
        ".pfm", ".png"
    )
    segmented_map = ((segmented_map / (n_clusters + 1)) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(segmented_img_file_path), exist_ok=True)
    cv2.imwrite(segmented_img_file_path, segmented_map)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process raw gaze data.")
    parser.add_argument(
        "--n-clusters",
        "-n",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="Number of clusters to segment the depth images into.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for segmenting depth images into 3 clusters: far, medium, and near.
    """
    args = parse_arguments()
    n_clusters = args.n_clusters

    # Get recursive list of all files in the directory
    file_paths = get_files_recursive(DEPTH_MAP_PFM_PATH, "*.pfm")
    for i, file_path in tqdm(
        enumerate(file_paths), desc="⌛ Segmenting depth images...", unit="image"
    ):
        segment_depth(file_path=file_path, n_clusters=n_clusters)


if __name__ == "__main__":
    main()
