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

from src.utils.eye_tracking_data import get_eye_tracking_data
from src.utils.saliency import get_kde_density
from src.config import SALIENCY_MAP_IMG_PATH, SALIENCY_MAP_PFM_PATH


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process raw gaze data.")
    parser.add_argument(
        "--saliency-width",
        "-sw",
        type=int,
        default=100,
        help="Width of the saliency map.",
    )
    parser.add_argument(
        "--saliency-height",
        "-sh",
        type=int,
        default=100,
        help="Height of the saliency map.",
    )
    parser.add_argument(
        "--bandwidth",
        "-b",
        type=float,
        default=5,
        help="Bandwidth of the kernel density estimation.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    saliency_width = args.saliency_width
    saliency_height = args.saliency_height
    bandwidth = args.bandwidth

    # Get fixation data
    fixation_data = get_eye_tracking_data(fixation=True)

    # Get KDE density for each sequence
    for experiment_id in fixation_data["ExperimentId"].unique():
        experiment_data = fixation_data[fixation_data["ExperimentId"] == experiment_id]
        for session_id in experiment_data["SessionId"].unique():
            session_data = experiment_data[experiment_data["SessionId"] == session_id]
            for sequence_id in tqdm(
                session_data["SequenceId"].unique(),
                desc=f"⌛ Computing KDE density for experiment {experiment_id} session {session_id}...",
                unit="sequence",
            ):
                sequence_data = session_data[
                    session_data["SequenceId"] == sequence_id
                ]
                coordinates = sequence_data[["X_sc", "Y_sc"]].values
                coordinates = np.array(
                    [
                        [int(coord[0] * saliency_width), int(coord[1] * saliency_height)]
                        for coord in coordinates
                    ]
                )
                kde_density = get_kde_density(
                    coordinates=coordinates,
                    saliency_width=saliency_width,
                    saliency_height=saliency_height,
                    kde_bandwidth=bandwidth,
                    apply_exponential=True,
                )
                kde_density = kde_density.astype(np.float32)

                # Save saliency map
                if experiment_id == 1:
                    session_str = "images" if session_id == 1 else "videos"
                else:
                    session_str = "clear" if session_id == 1 else "overcast"
                saliency_map_pfm_path = (
                    f"{SALIENCY_MAP_PFM_PATH}/experiment{experiment_id}/{session_str}/scene{sequence_id}.pfm"
                )
                os.makedirs(os.path.dirname(saliency_map_pfm_path), exist_ok=True)
                justpfm.write_pfm(file_name=saliency_map_pfm_path, data=kde_density)
                saliency_map_img_path = (
                    f"{SALIENCY_MAP_IMG_PATH}/experiment{experiment_id}/{session_str}/scene{sequence_id}.png"
                )
                kde_density = cv2.normalize(
                    kde_density, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                os.makedirs(os.path.dirname(saliency_map_img_path), exist_ok=True)
                cv2.imwrite(saliency_map_img_path, kde_density)


if __name__ == "__main__":
    main()
