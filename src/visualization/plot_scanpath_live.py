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

from src.utils.coordinates_buffer import CoordinatesBuffer
from src.visualization.plot_eye_tracking import (
    get_grouped_processed_data,
    get_grouped_fixation_data,
    get_background,
    get_background_frame,
    draw_information,
)
from src.config import GENERATED_PATH


COLORMAP = cv2.COLORMAP_JET
DEFAULT_FRAME_WIDTH = 768
DEFAULT_FRAME_HEIGHT = 384
DEFAULT_FPS = 60
DEFAULT_CIRCLE_RADIUS = 3
DEFAULT_LINE_THICKNESS = 1
DEFAULT_TRAIL_LENGTH = 1


def update_coordinates_buffers(
    groups: List[pd.DataFrame],
    next_frames: List[int],
    curr_frame: int,
    max_frame: int,
    coordinates_buffers: List[CoordinatesBuffer],
) -> Tuple[List[CoordinatesBuffer], List[int]]:
    """
    Update coordinates buffers with current gaze coordinates.

    Args:
        groups (List[pd.DataFrame]): The eye tracking data grouped by single sequence experiment.
        next_frames (List[int]): The next frame for each group.
        curr_frame (int): The current frame number.
        max_frame (int): The maximum frame number.
        coordinates_buffers (List[CoordinatesBuffer]): The coordinates buffers.

    Returns:
        List[CoordinatesBuffer]: The updated coordinates buffers.
    """
    for i, group in enumerate(groups):
        next_frame = next_frames[i]
        if curr_frame == next_frame:
            # Update current gaze coordinates
            curr_group_coordinates = group[group["FrameNumber"] == curr_frame][
                ["X_sc", "Y_sc"]
            ].values[0]
            coordinates_buffers[i].add(curr_group_coordinates)

            # Update next frame
            next_frame = group[group["FrameNumber"] > curr_frame]["FrameNumber"]
            if len(next_frame) > 0:
                next_frame = next_frame.iloc[0]
            else:
                next_frame = (
                    max_frame + 1
                )  # Set to max frame to avoid updating gaze coordinates
            next_frames[i] = next_frame

    return coordinates_buffers, next_frames


def get_fixations(
    groups: List[pd.DataFrame],
    curr_frame: int,
) -> List[Tuple[float, float]]:
    """
    Get fixations for the current frame.

    Args:
        groups (List[pd.DataFrame]): The fixation data grouped by single sequence experiment.
        curr_frame (int): The current frame number.

    Returns:
        List[Tuple[float, float]]: The fixations for the current frame.
    """
    fixations = []
    for group in groups:
        group_fixations = group[(group["StartFrameNumber"] <= curr_frame) & (curr_frame <= group["EndFrameNumber"])][
            ["X_sc", "Y_sc"]
        ].values
        fixations.extend(group_fixations)

    return fixations


def draw_gaze_scanpath_live(
    coordinates_buffers: List[CoordinatesBuffer],
    fixations: List[Tuple[float, float]],
    frame: np.ndarray,
    frame_width: int,
    frame_height: int,
    circle_radius: int,
    line_thickness: int,
) -> np.ndarray:
    """
    Draw gaze live sequence on the frame.

    Args:
        coordinates_buffers (List[CoordinatesBuffer]): The coordinates buffers.
        fixations (List[Tuple[float, float]]): The fixations.
        frame (np.ndarray): The frame.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        circle_radius (int): The circle radius.
        line_thickness (int): The line thickness.

    Returns:
        np.ndarray: The frame with the gaze sequence drawn.
    """
    # Get colors to use for each participant
    values = np.linspace(0, 255, len(coordinates_buffers)).astype(np.uint8)
    colormap = cv2.applyColorMap(values[:, None], COLORMAP)
    colors = colormap[:, 0, :]

    overlay = np.zeros_like(frame)
    for i, coordinate_buffer in enumerate(coordinates_buffers):
        if coordinate_buffer.is_empty():
            continue

        # Draw last point in buffer
        x, y = coordinate_buffer.get_most_recent()
        x = int(x * frame_width)
        y = int(y * frame_height)
        color = colors[i].tolist()
        cv2.circle(
            overlay,
            (x, y),
            radius=circle_radius,
            color=color,
            thickness=-1,
        )

        # Draw scanpath trail
        for (x1, y1), (x2, y2) in zip(coordinate_buffer, coordinate_buffer[1:]):
            x1 = int(x1 * frame_width)
            y1 = int(y1 * frame_height)
            x2 = int(x2 * frame_width)
            y2 = int(y2 * frame_height)
            cv2.line(
                overlay,
                (x1, y1),
                (x2, y2),
                color=color,
                thickness=line_thickness,
            )

        # Draw fixations
        for fixation in fixations:
            x, y = fixation
            x = int(x * frame_width)
            y = int(y * frame_height)
            cv2.circle(
                overlay,
                (x, y),
                radius=circle_radius,
                color=(255, 255, 255),
                thickness=-1,
            )

    overlay_mask = overlay.sum(axis=2) > 0
    frame[overlay_mask] = overlay[overlay_mask]

    return frame


def visualize_gaze_scanpath_live(
    experiment_id: int,
    session_id: int,
    participant_ids: List[int] | None,
    sequence_id: int,
    output_file_path: str,
    frame_width: int,
    frame_height: int,
    fps: int,
    circle_radius: int,
    line_thickness: int,
    trail_length: int,
    use_interpolated: bool,
) -> None:
    """
    Visualize gaze live sequence for the given experiment, session, participant(s), and sequence.

    Args:
        experiment_id (int): The experiment ID.
        session_id (int): The session ID.
        participant_ids (List[int] | None): The participant IDs.
        sequence_id (int): The sequence ID.
        output_file_path (str): The output file path.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        fps (int): The frames per second.
        circle_radius (int): The circle radius.
        line_thickness (int): The line thickness.
        trail_length (int): The trail length.
        use_interpolated (bool): Whether to use interpolated data.
    """
    # Get eye tracking data
    processed_groups = get_grouped_processed_data(
        experiment_id=experiment_id,
        session_id=session_id,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        fps=fps,
        interpolated=use_interpolated,
    )

    fixation_groups = get_grouped_fixation_data(
        experiment_id=experiment_id,
        session_id=session_id,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        fps=fps,
        processed_groups=processed_groups,
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
    next_frames = [group["FrameNumber"].iloc[0] for group in processed_groups]
    max_frame = max([group["FrameNumber"].max() for group in processed_groups])
    coordinates_buffers = [CoordinatesBuffer(max_length=trail_length) for _ in processed_groups]
    fixations = []
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
        coordinates_buffers, next_frames = update_coordinates_buffers(
            groups=processed_groups,
            next_frames=next_frames,
            curr_frame=curr_frame,
            max_frame=max_frame,
            coordinates_buffers=coordinates_buffers,
        )

        fixations = get_fixations(
            groups=fixation_groups,
            curr_frame=curr_frame,
        )

        # Draw gaze coordinates and information on frame
        frame = draw_gaze_scanpath_live(
            coordinates_buffers=coordinates_buffers,
            fixations=fixations,
            frame=frame,
            frame_width=frame_width,
            frame_height=frame_height,
            circle_radius=circle_radius,
            line_thickness=line_thickness,
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

    out.release()
    cv2.destroyAllWindows()
    bar.close()

    print("✅ Live scanpath plot generated.")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: The command line arguments.
    """
    parser = argparse.ArgumentParser(description="Visualize gaze sequence.")
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
        "--participant-ids",
        "-p",
        type=int,
        nargs="+",
        default=None,
        help="The participant IDs.",
    )
    parser.add_argument(
        "--sequence-id",
        "-sq",
        type=int,
        required=True,
        help="The sequence ID.",
    )
    parser.add_argument(
        "--output-file-path",
        "-out",
        type=str,
        default=f"{GENERATED_PATH}/scanpath_live.mp4",
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
        "--circle_radius",
        "-r",
        type=int,
        default=DEFAULT_CIRCLE_RADIUS,
        help="The circle radius.",
    )
    parser.add_argument(
        "--line_thickness",
        "-t",
        type=int,
        default=DEFAULT_LINE_THICKNESS,
        help="The line thickness.",
    )
    parser.add_argument(
        "--trail_length",
        "-l",
        type=int,
        default=DEFAULT_TRAIL_LENGTH,
        help="The trail length.",
    )
    parser.add_argument(
        "--use-interpolated",
        "-i",
        action="store_true",
        help="Whether to use interpolated data.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for visualizing gaze data as a live sequence showing the gaze scanpath.
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
    circle_radius = args.circle_radius
    line_thickness = args.line_thickness
    trail_length = args.trail_length
    use_interpolated = args.use_interpolated

    visualize_gaze_scanpath_live(
        experiment_id=experiment_id,
        session_id=session_id,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        output_file_path=output_file_path,
        frame_width=frame_width,
        frame_height=frame_height,
        fps=fps,
        circle_radius=circle_radius,
        line_thickness=line_thickness,
        trail_length=trail_length,
        use_interpolated=use_interpolated,
    )


if __name__ == "__main__":
    main()
