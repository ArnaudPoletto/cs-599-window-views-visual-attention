import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import argparse
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt

from src.visualization.plot_eye_tracking import (
    get_grouped_eye_tracking_data,
    get_background,
    draw_gaze_saliency,
)
from src.config import GENERATED_PATH

DEFAULT_FRAME_WIDTH = 768
DEFAULT_FRAME_HEIGHT = 384
SALIENCY_RESOLUTION_RATIO = 0.25
DEFAULT_KDE_BANDWIDTH = 10.0


def get_coordinates(groups: List[pd.DataFrame]) -> List[Tuple[float, float]]:
    """
    Get the gaze coordinates.

    Args:
        groups (List[pd.DataFrame]): The grouped eye tracking data.

    Returns:
        List[Tuple[float, float]]: The gaze coordinates.
    """
    coordinates = []
    for group in groups:
        group_coordinates = group[["GazeX", "GazeY"]].values
        if group_coordinates.size > 0:
            coordinates.extend(group_coordinates)
    return coordinates

def get_gaze_saliency_global(
    experiment_id: int,
    session_id: int,
    participant_ids: List[int] | None,
    sequence_id: int,
    frame_width: int,
    frame_height: int,
    saliency_resolution_ratio: float,
    kde_bandwidth: float,
):
    # Get eye tracking data
    groups = get_grouped_eye_tracking_data(
        experiment_id=experiment_id,
        session_id=session_id,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        fps=1,
    )

    # Get background image, taking first frame if this is a video sequence
    background, _ = get_background(
        experiment_id=experiment_id,
        session_id=session_id,
        sequence_id=sequence_id,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    if isinstance(background, List):
        background = background[0]

    # Get coordinates for each session
    coordinates = get_coordinates(groups=groups)

    # Draw gaze saliency for each session
    saliency_width = int(frame_width * saliency_resolution_ratio)
    saliency_height = int(frame_height * saliency_resolution_ratio)
    frame = draw_gaze_saliency(
        coordinates=coordinates,
        frame=background,
        frame_width=frame_width,
        frame_height=frame_height,
        saliency_width=saliency_width,
        saliency_height=saliency_height,
        kde_bandwidth=kde_bandwidth,
    )

    return frame

def visualize_gaze_saliency_global(
    experiment_id: int,
    participant_ids: List[int] | None,
    sequence_id: int,
    output_file_path: str,
    frame_width: int,
    frame_height: int,
    saliency_resolution_ratio: float,
    kde_bandwidth: float,
) -> None:
    """
    Visualize gaze saliency for the given experiment, session, participant(s), and sequence.

    Args:
        experiment_id (int): The experiment ID.
        participant_id (int): The participant ID.
        sequence_id (int): The sequence ID.
        output_file_path (str): The output file path.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        saliency_resolution_ratio (float): The saliency resolution ratio.
        kde_bandwidth (float): The bandwidth for the Kernel Density Estimation.
    """
    session1_frame = get_gaze_saliency_global(
        experiment_id=experiment_id,
        session_id=1,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        frame_width=frame_width,
        frame_height=frame_height,
        saliency_resolution_ratio=saliency_resolution_ratio,
        kde_bandwidth=kde_bandwidth,
    )
    session1_frame = session1_frame[..., ::-1]
    session2_frame = get_gaze_saliency_global(
        experiment_id=experiment_id,
        session_id=2,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        frame_width=frame_width,
        frame_height=frame_height,
        saliency_resolution_ratio=saliency_resolution_ratio,
        kde_bandwidth=kde_bandwidth,
    )
    session2_frame = session2_frame[..., ::-1]

    plt.figure(figsize=(13, 4))
    plt.suptitle(f"Global saliency map of experiment {experiment_id}, sequence {sequence_id}{f", participants {" ".join([str(pid) for pid in participant_ids])}" if participant_ids else ''}")
    plt.subplot(1, 2, 1)
    plt.imshow(session1_frame)
    plt.title("Session 1")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(session2_frame)
    plt.title("Session 2")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.close()

    print("✅ Global saliency plot generated.")

def parse_arguments() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize gaze saliency.")
    parser.add_argument(
        "--experiment-id",
        "-e",
        type=int,
        required=True,
        help="The experiment ID.",
    )
    parser.add_argument(
        "--sequence-id",
        "-sq",
        type=int,
        required=True,
        help="The sequence ID.",
    )
    parser.add_argument(
        "--participant-ids",
        "-p",
        type=int,
        nargs="+",
        default=None,
        help="The participant IDs.",
    )
    parser.add_argument(
        "--output-file-path",
        "-out",
        type=str,
        default=f"{GENERATED_PATH}/saliency_global.png",
        help="The output file path.",
    )
    parser.add_argument(
        "--frame-width",
        "-fw",
        type=int,
        default=DEFAULT_FRAME_WIDTH,
        help="The frame width.",
    )
    parser.add_argument(
        "--frame_height",
        "-fh",
        type=int,
        default=DEFAULT_FRAME_HEIGHT,
        help="The frame height.",
    )
    parser.add_argument(
        "--saliency-resolution-ratio",
        "-sr",
        type=float,
        default=SALIENCY_RESOLUTION_RATIO,
        help="The saliency resolution ratio.",
    )
    parser.add_argument(
        "--kde-bandwidth",
        "-kdeb",
        type=float,
        default=DEFAULT_KDE_BANDWIDTH,
        help="The bandwidth for the Kernel Density Estimation.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for visualizing eye tracking data as a global saliency map showing all collected gaze points.
    """
    args = parse_arguments()
    experiment_id = args.experiment_id
    participant_ids = args.participant_ids
    sequence_id = args.sequence_id
    output_file_path = args.output_file_path
    frame_width = args.frame_width
    frame_height = args.frame_height
    saliency_resolution_ratio = args.saliency_resolution_ratio
    kde_bandwidth = args.kde_bandwidth

    visualize_gaze_saliency_global(
        experiment_id=experiment_id,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        output_file_path=output_file_path,
        frame_width=frame_width,
        frame_height=frame_height,
        saliency_resolution_ratio=saliency_resolution_ratio,
        kde_bandwidth=kde_bandwidth,
    )


if __name__ == "__main__":
    main()
