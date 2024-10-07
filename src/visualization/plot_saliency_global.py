import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import argparse
import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt

from src.visualization.plot_eye_tracking import (
    get_grouped_processed_data,
    get_grouped_fixation_data,
    get_background,
    draw_saliency,
)
from src.config import GENERATED_PATH

DEFAULT_FRAME_WIDTH = 768
DEFAULT_FRAME_HEIGHT = 384
SALIENCY_RESOLUTION_RATIO = 0.25
DEFAULT_KDE_BANDWIDTH = 10.0


def get_coordinates(groups: List[pd.DataFrame]) -> List[Tuple[float, float]]:
    """
    Get coordinates from given groups.

    Args:
        groups (List[pd.DataFrame]): The grouped data.

    Returns:
        List[Tuple[float, float]]: The coordinates.
    """
    coordinates = []
    for group in groups:
        group_coordinates = group[["X_sc", "Y_sc"]].values
        coordinates.extend(group_coordinates)
    return coordinates


def get_saliency_global(
    experiment_ids: List[int] | None,
    session_ids: List[int] | None,
    participant_ids: List[int] | None,
    sequence_ids: List[int] | None,
    set_ids: List[int] | None,
    frame_width: int,
    frame_height: int,
    saliency_resolution_ratio: float,
    kde_bandwidth: float,
    use_fixations: bool,
    use_interpolated: bool,
    n_samples: int | None = None,
):
    """
    Get global saliency map for specified data.

    Args:
        experiment_ids (List[int] | None): The experiment ID.
        session_ids (List[int] | None): The session ID.
        participant_ids (List[int] | None): The participant ID.
        sequence_ids (List[int] | None): The sequence ID.
        set_ids (List[int] | None): The set ID.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        saliency_resolution_ratio (float): The saliency resolution ratio, i.e. the downscaling factor for the saliency map from the frame.
        kde_bandwidth (float): The bandwidth for the Kernel Density Estimation.
        use_fixations (bool): Whether to use fixations instead of gaze points.
        use_interpolated (bool): Whether to use interpolated data.
        n_samples (int | None): The number of samples to use.

    Returns:
        np.ndarray: The frame with the saliency map.
    """
    # Get data
    processed_groups = get_grouped_processed_data(
        experiment_ids=experiment_ids,
        session_ids=session_ids,
        participant_ids=participant_ids,
        sequence_ids=sequence_ids,
        set_ids=set_ids,
        fps=1,
        interpolated=use_interpolated,
    )
    if use_fixations:
        groups = get_grouped_fixation_data(
            experiment_ids=experiment_ids,
            session_ids=session_ids,
            participant_ids=participant_ids,
            sequence_ids=sequence_ids,
            set_ids=set_ids,
            fps=1,
            processed_groups=processed_groups,
        )
    else:
        groups = processed_groups

    # Get background image, taking first frame if this is a video sequence
    # If multiple experiments, sessions, or sequences are provided, use a black background
    if (
        experiment_ids is not None
        and len(experiment_ids) == 1
        and sequence_ids is not None
        and len(sequence_ids) == 1
        and set_ids is not None
        and len(set_ids) == 1
    ):
        experiment_id = experiment_ids[0]
        sequence_id = sequence_ids[0]
        set_id = set_ids[0]
        background, _ = get_background(
            experiment_id=experiment_id,
            sequence_id=sequence_id,
            set_id=set_id,
            frame_width=frame_width,
            frame_height=frame_height,
            only_first_frame=True,
        )
    else:
        background = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Get coordinates for each session
    coordinates = get_coordinates(groups=groups)

    # Sample coordinates
    if n_samples is not None:
        n_samples = min(n_samples, len(coordinates))
        coordinates = np.array(coordinates)
        idx = np.random.choice(len(coordinates), n_samples, replace=False)
        coordinates = coordinates[idx]
        coordinates = list(map(tuple, coordinates))

    # Draw saliency
    saliency_width = int(frame_width * saliency_resolution_ratio)
    saliency_height = int(frame_height * saliency_resolution_ratio)
    frame = draw_saliency(
        coordinates=coordinates,
        frame=background,
        frame_width=frame_width,
        frame_height=frame_height,
        saliency_width=saliency_width,
        saliency_height=saliency_height,
        kde_bandwidth=kde_bandwidth,
    )

    return frame


def visualize_saliency_global(
    experiment_ids: List[int] | None,
    session_ids: List[int] | None,
    participant_ids: List[int] | None,
    sequence_ids: List[int] | None,
    set_ids: List[int] | None,
    output_file_path: str,
    frame_width: int,
    frame_height: int,
    saliency_resolution_ratio: float,
    kde_bandwidth: float,
    use_fixations: bool,
    use_interpolated: bool,
    n_samples: int | None = None,
) -> None:
    """
    Visualize saliency.

    Args:
        experiment_ids (List[int] | None): The experiment ID.
        session_ids (List[int] | None): The session ID.
        participant_ids (List[int] | None): The participant ID.
        sequence_ids (List[int] | None): The sequence ID.
        set_ids (List[int] | None): The set ID.
        output_file_path (str): The output file path.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        saliency_resolution_ratio (float): The saliency resolution ratio.
        kde_bandwidth (float): The bandwidth for the Kernel Density Estimation.
        use_fixations (bool): Use fixations instead of gaze points.
        use_interpolated (bool): Whether to use interpolated data.
        n_samples (int | None): The number of samples to use.
    """
    if use_fixations and use_interpolated:
        raise ValueError("❌ Cannot use both fixations and interpolated data.")

    frame = get_saliency_global(
        experiment_ids=experiment_ids,
        session_ids=session_ids,
        participant_ids=participant_ids,
        sequence_ids=sequence_ids,
        set_ids=set_ids,
        frame_width=frame_width,
        frame_height=frame_height,
        saliency_resolution_ratio=saliency_resolution_ratio,
        kde_bandwidth=kde_bandwidth,
        use_fixations=use_fixations,
        use_interpolated=use_interpolated,
        n_samples=n_samples,
    )
    frame = frame[..., ::-1]

    plt.figure(figsize=(10, 5))
    participants_str = (
        "all participants"
        if participant_ids is None
        else f"participant(s) {', '.join(map(str, participant_ids))}"
    )
    experiment_str = (
        f"all experiments"
        if experiment_ids is None
        else f"experiment(s) {', '.join(map(str, experiment_ids))}"
    )
    session_str = (
        f"all sessions"
        if session_ids is None
        else f"session(s) {', '.join(map(str, session_ids))}"
    )
    sequence_str = (
        f"all sequences"
        if sequence_ids is None
        else f"sequence(s) {', '.join(map(str, sequence_ids))}"
    )
    set_str = (
        f"all sets" if set_ids is None else f"set(s) {', '.join(map(str, set_ids))}"
    )
    plt.suptitle(
        f"Global saliency map of {participants_str} for {', '.join([experiment_str, session_str, sequence_str, set_str])}",
    )
    plt.imshow(frame)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file_path)
    plt.close()

    print("✅ Global saliency plot generated.")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize global saliency map.")
    parser.add_argument(
        "--experiment-ids",
        "-e",
        type=int,
        nargs="+",
        default=None,
        help="The experiment ID.",
    )
    parser.add_argument(
        "--session-ids",
        "-se",
        type=int,
        nargs="+",
        default=None,
        help="The session ID.",
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
        "--sequence-ids",
        "-sq",
        type=int,
        nargs="+",
        default=None,
        help="The sequence ID.",
    )
    parser.add_argument(
        "--set-ids",
        "-s",
        type=int,
        nargs="+",
        default=None,
        help="The set IDs.",
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
        "-kb",
        type=float,
        default=DEFAULT_KDE_BANDWIDTH,
        help="The bandwidth for the Kernel Density Estimation.",
    )
    parser.add_argument(
        "--use-fixations",
        "-f",
        action="store_true",
        help="Whether to use fixations instead of gaze points.",
    )
    parser.add_argument(
        "--use-interpolated",
        "-i",
        action="store_true",
        help="Whether to use interpolated data.",
    )
    parser.add_argument(
        "--n-samples",
        "-n",
        type=int,
        default=None,
        help="The number of samples to use.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for visualizing data as a global saliency map showing all collected points.
    """
    args = parse_arguments()
    experiment_ids = args.experiment_ids
    session_ids = args.session_ids
    participant_ids = args.participant_ids
    sequence_ids = args.sequence_ids
    set_ids = args.set_ids
    output_file_path = args.output_file_path
    frame_width = args.frame_width
    frame_height = args.frame_height
    saliency_resolution_ratio = args.saliency_resolution_ratio
    kde_bandwidth = args.kde_bandwidth
    use_fixations = args.use_fixations
    use_interpolated = args.use_interpolated
    n_samples = args.n_samples

    visualize_saliency_global(
        experiment_ids=experiment_ids,
        session_ids=session_ids,
        participant_ids=participant_ids,
        sequence_ids=sequence_ids,
        set_ids=set_ids,
        output_file_path=output_file_path,
        frame_width=frame_width,
        frame_height=frame_height,
        saliency_resolution_ratio=saliency_resolution_ratio,
        kde_bandwidth=kde_bandwidth,
        use_fixations=use_fixations,
        use_interpolated=use_interpolated,
        n_samples=n_samples,
    )


if __name__ == "__main__":
    main()
