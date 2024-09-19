import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import cv2
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
from sklearn.neighbors import KernelDensity

from src.visualization.plot_eye_tracking import (
    get_grouped_eye_tracking_data,
    get_background,
    get_max_frame,
    get_background_frame,
    draw_information,
)
from src.config import GENERATED_PATH

END_BACKGROUND_DARK_RATIO = 0.5
COLORMAP = cv2.COLORMAP_HOT
DEFAULT_FRAME_WIDTH = 768
DEFAULT_FRAME_HEIGHT = 384
DEFAULT_FPS = 60
SALIENCY_RESOLUTION_RATIO = 0.25
DEFAULT_KDE_BANDWIDTH = 10.0


def __update_group_coordinates(
    groups: List[pd.DataFrame],
    next_frames: List[int],
    curr_frame: int,
    max_frame: int,
    group_coordinates: List[Tuple[float, float]],
) -> Tuple[List[Tuple[float, float]], List[int]]:
    """
    Get the current gaze coordinates.
    
    Args:
        groups (List[pd.DataFrame]): The grouped eye tracking data.
        next_frames (List[int]): The next frames.
        curr_frame (int): The current frame.
        max_frame (int): The maximum frame.
        coordinates (List[Tuple[float, float]]): The current gaze coordinates.
        
    Returns:
        List[Tuple[float, float]]: The current gaze coordinates.
    """
    for i, group in enumerate(groups):
        next_frame = next_frames[i]
        if curr_frame == next_frame:
            # Update current gaze coordinates
            curr_group_coordinates = group[group["FrameNumber"] == curr_frame][["GazeX", "GazeY"]].values
            if curr_group_coordinates.size > 0:
                group_coordinates[i] = curr_group_coordinates

            # Update next frame
            next_frame = group[group["FrameNumber"] > curr_frame]["FrameNumber"]
            if len(next_frame) > 0:
                next_frame = next_frame.iloc[0]
            else:
                next_frame = (
                    max_frame + 1
                )  # Set to max frame to avoid updating gaze coordinates
            next_frames[i] = next_frame

    return group_coordinates, next_frames

def __draw_gaze_saliency(
    coordinates: List[Tuple[float, float]],
    frame: np.ndarray,
    frame_width: int,
    frame_height: int,
    saliency_width: int,
    saliency_height: int,
    kde_bandwidth: float
) -> np.ndarray:
    """
    Draw gaze saliency on frame.

    Args:
        coordinates (List[Tuple[float, float]]): The gaze coordinates.
        frame (np.ndarray): The frame.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        saliency_width (int): The saliency map width.
        saliency_height (int): The saliency map height.
        kde_bandwidth (float): The bandwidth for the Kernel Density Estimation.

    Returns:
        np.ndarray: The frame with gaze saliency.
    """
    # Return frame if no gaze coordinates
    if not coordinates:
        return frame
    
    # Rescale gaze coordinates to frame size
    coordinates = np.array([
        [int(coord[0] * saliency_width), int(coord[1] * saliency_height)]
        for coord in coordinates
    ])

    # Fit Kernel Density Estimation to gaze coordinates
    x_grid, y_grid = np.meshgrid(
        np.linspace(0, saliency_width - 1, saliency_width),
        np.linspace(0, saliency_height - 1, saliency_height),
    )
    grid_sample = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    kde = KernelDensity(bandwidth=kde_bandwidth, kernel="gaussian").fit(coordinates)
    kde.fit(coordinates)
    kde_scores = kde.score_samples(grid_sample)
    kde_density = np.exp(kde_scores).reshape(saliency_height, saliency_width)
    kde_density = cv2.normalize(kde_density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply colormap to estimated density, resize saliency map and mix with frame
    saliency_map = cv2.applyColorMap(kde_density, COLORMAP)
    saliency_map = cv2.resize(saliency_map, (frame_width, frame_height))
    frame = cv2.addWeighted(frame, 0.5, saliency_map, 0.5, 0)

    return frame


def __visualize_gaze_saliency(
    experiment_id: int,
    session_id: int,
    participant_ids: int | None,
    sequence_id: int,
    output_file_path: str,
    frame_width: int,
    frame_height: int,
    fps: int,
    saliency_resolution_ratio: float,
    kde_bandwidth: float,
    end_at_video_end: bool,
) -> None:
    """
    Visualize gaze saliency for the given experiment, session, participant(s), and sequence.
    
    Args:
        experiment_id (int): The experiment ID.
        session_id (int): The session ID.
        participant_ids (int | None): The participant IDs.
        sequence_id (int): The sequence ID.
        output_file_path (str): The output file path.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        fps (int): The frames per second.
        saliency_resolution_ratio (float): The saliency resolution ratio.
        kde_bandwidth (float): The bandwidth for the Kernel Density Estimation.
        end_at_video_end (bool): Whether to end gaze visualization at the end of the video.
    """
    # Get eye tracking data
    groups = get_grouped_eye_tracking_data(
        experiment_id=experiment_id,
        session_id=session_id,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        fps=fps,
    )

    # Get background image or video
    background, background_fps = get_background(
        experiment_id=experiment_id,
        session_id=session_id,
        sequence_id=sequence_id,
        frame_width=frame_width,
        frame_height=frame_height,
    )

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc("a", "v", "c", "1")
    out = cv2.VideoWriter(output_file_path, fourcc, fps, (frame_width, frame_height))

    curr_frame = 0
    next_frames = [group["FrameNumber"].iloc[0] for group in groups]
    max_frame = get_max_frame(groups=groups, background=background, end_at_video_end=end_at_video_end)
    group_coordinates = [[] for _ in groups]
    bar = tqdm(total=max_frame, desc="⌛ Generating gaze video...", unit="frames")
    while curr_frame < max_frame:

        # Get current background frame
        frame = get_background_frame(
            background=background,
            curr_frame=curr_frame,
            background_fps=background_fps,
            fps=fps,
        )

        # Update current gaze coordinates
        group_coordinates, next_frames = __update_group_coordinates(
            groups=groups,
            next_frames=next_frames,
            curr_frame=curr_frame,
            max_frame=max_frame,
            group_coordinates=group_coordinates,
        )

        # Draw gaze coordinates and information on frame
        coordinates = [coord for group_coords in group_coordinates for coord in group_coords]
        saliency_width = int(frame_width * saliency_resolution_ratio)
        saliency_height = int(frame_height * saliency_resolution_ratio)
        frame = __draw_gaze_saliency(
            coordinates=coordinates,
            frame=frame,
            frame_width=frame_width,
            frame_height=frame_height,
            saliency_width=saliency_width,
            saliency_height=saliency_height,
            kde_bandwidth=kde_bandwidth,
        )

        frame = draw_information(
            frame=frame,
            curr_frame=curr_frame,
            max_frame=max_frame,
            experiment_id=experiment_id,
            session_id=session_id,
            sequence_id=sequence_id,
            frame_width=frame_width,
        )

        out.write(frame)
        curr_frame += 1
        bar.update(1)

    if end_at_video_end and isinstance(background, List):
        print("✅ Gaze video generated, ending at video end.")
    else:
        print("✅ Gaze video generated, ending at eye tracking data end.")

    out.release()
    cv2.destroyAllWindows()
    bar.close()


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
        "--session-id",
        "-se",
        type=int,
        required=True,
        help="The session ID.",
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
        default=f"{GENERATED_PATH}/gaze_saliency.mp4",
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
        "--fps",
        type=int,
        default=DEFAULT_FPS,
        help="The frames per second.",
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
    parser.add_argument(
        "--end-at-video-end",
        action="store_true",
        help="End gaze visualization at the end of the video.",
    )

    return parser.parse_args()

def main() -> None:
    """
    Main function for visualizing eye tracking sequence.
    """
    args = parse_arguments()
    experiment_id = args.experiment_id
    session_id = args.session_id
    participant_ids = args.participant_ids
    sequence_id = args.sequence_id
    output_file_path = args.output_file_path
    frame_width = args.frame_width
    frame_height = args.frame_height
    fps = args.fps
    saliency_resolution_ratio = args.saliency_resolution_ratio
    kde_bandwidth = args.kde_bandwidth
    end_at_video_end = args.end_at_video_end

    __visualize_gaze_saliency(
        experiment_id=experiment_id,
        session_id=session_id,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        output_file_path=output_file_path, 
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        saliency_resolution_ratio=saliency_resolution_ratio,
        kde_bandwidth=kde_bandwidth,
        end_at_video_end=end_at_video_end,
    )

if __name__ == "__main__":
    main()