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

from src.utils.eye_tracking_data import get_gaze_data
from src.utils.kde import get_kde_density
from src.utils.file import get_set_str
from src.config import SALIENCY_MAP_IMG_PATH, SALIENCY_MAP_PFM_PATH, IMAGE_WIDTH, IMAGE_HEIGHT

KDE_WIDTH = 192
KDE_HEIGHT = 96
KDE_BANDWIDTH = 10
N_SAMPLES = -1


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Get global saliency maps.")
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
        "--kde_bandwidth",
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
    Main function for computing the global saliency maps for each sequence.
    """
    args = parse_arguments()
    kde_width = args.kde_width
    kde_height = args.kde_height
    kde_bandwidth = args.kde_bandwidth
    n_samples = args.n_samples

    # Get fixation data
    fixation_data = get_gaze_data(fixation=True)

    # Get KDE density for each sequence
    for experiment_id in fixation_data["ExperimentId"].unique():
        experiment_data = fixation_data[fixation_data["ExperimentId"] == experiment_id]
        for set_id in experiment_data["SetId"].unique():
            set_data = experiment_data[experiment_data["SetId"] == set_id]
            for sequence_id in tqdm(
                set_data["SequenceId"].unique(),
                desc=f"⌛ Computing global saliency maps for Experiment {experiment_id} Set {set_id}...",
                unit="sequence",
            ):
                sequence_data = set_data[set_data["SequenceId"] == sequence_id]
                coordinates = sequence_data[["X_sc", "Y_sc"]].values
                coordinates = np.array(
                    [
                        [int(coord[0] * kde_width), int(coord[1] * kde_height)]
                        for coord in coordinates
                    ]
                )
                kde_density = get_kde_density(
                    coordinates=coordinates,
                    kde_width=kde_width,
                    kde_height=kde_height,
                    kde_bandwidth=kde_bandwidth,
                    n_samples=n_samples,
                    apply_exponential=True,
                )
                kde_density = cv2.resize(kde_density, (IMAGE_WIDTH, IMAGE_HEIGHT))
                kde_density = kde_density.astype(np.float32)

                # Save saliency map
                set_str = get_set_str(experiment_id, set_id)
                saliency_map_pfm_path = f"{SALIENCY_MAP_PFM_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}.pfm"
                os.makedirs(os.path.dirname(saliency_map_pfm_path), exist_ok=True)
                justpfm.write_pfm(file_name=saliency_map_pfm_path, data=kde_density)
                
                saliency_map_img_path = f"{SALIENCY_MAP_IMG_PATH}/experiment{experiment_id}/{set_str}/scene{sequence_id:02}.png"
                kde_density = cv2.normalize(
                    kde_density, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
                os.makedirs(os.path.dirname(saliency_map_img_path), exist_ok=True)
                cv2.imwrite(saliency_map_img_path, kde_density)

    print(f"✅ Global saliency maps computed and saved at {Path(SALIENCY_MAP_PFM_PATH).resolve()}.")


if __name__ == "__main__":
    main()
