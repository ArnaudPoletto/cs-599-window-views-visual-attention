import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from justpfm import justpfm
from sklearn.cluster import KMeans

from src.utils.file import get_files_recursive
from src.config import (
    DEPTH_MAP_PFM_PATH,
    DEPTH_SEG_IMG_PATH,
    DEPTH_SEG_PFM_PATH,
)


def segment_depth(
    file_path: str,
    n_clusters: int,
) -> None:
    """
    Segment the given pfm depth image into 3 clusters: far, medium, and near.

    Args:
        file_path (str): Path to the pfm depth image file
        n_clusters (int): Number of clusters to segment the depth images into
    """
    # Load pfm file
    depth_map = justpfm.read_pfm(file_name=file_path)

    # Segment depth map into 3 segments using k-means clustering
    # 0: far, 1: medium, 2: near
    depth_values = depth_map.flatten().reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(depth_values)
    clustered_map = kmeans.labels_.reshape(depth_map.shape)
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorted_indices = np.argsort(cluster_centers)
    segmented_map = np.zeros_like(clustered_map)
    for i, sorted_idx in enumerate(sorted_indices):
        segmented_map[clustered_map == sorted_idx] = i
    segmented_map = segmented_map.astype(np.float32)

    # Save segmented depth map as pfm and image files
    segmented_pfm_file_path = file_path.replace("map_pfm", "seg_pfm")
    os.makedirs(os.path.dirname(segmented_pfm_file_path), exist_ok=True)
    justpfm.write_pfm(file_name=segmented_pfm_file_path, data=segmented_map)

    segmented_img_file_path = file_path.replace("map_pfm", "seg_png").replace(".pfm", ".png")
    segmented_map = ((segmented_map / (n_clusters - 1)) * 255).astype(np.uint8)
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
        default=3,
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
    for file_path in tqdm(
        file_paths, desc="⌛ Segmenting depth images...", unit="image"
    ):
        segment_depth(file_path=file_path, n_clusters=n_clusters)


if __name__ == "__main__":
    main()
