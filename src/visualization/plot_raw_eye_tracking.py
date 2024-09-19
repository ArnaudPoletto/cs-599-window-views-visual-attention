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
from src.utils.eye_tracking_data import get_eye_tracking_data
from src.config import SETS_PATH, GENERATED_PATH


N_HNANOSECONDS_IN_SECONDS = 1e7  # Number of hundred nanoseconds in a second
END_BACKGROUND_DARK_RATIO = 0.5
COLORMAP = cv2.COLORMAP_JET
DEFAULT_FRAME_WIDTH = 768
DEFAULT_FRAME_HEIGHT = 384
DEFAULT_FPS = 60
DEFAULT_CIRCLE_RADIUS = 3
DEFAULT_LINE_THICKNESS = 1
DEFAULT_TRAIL_LENGTH = 10


def __get_grouped_eye_tracking_data(
    experiment_id: int,
    session_id: int,
    participant_ids: int | None,
    sequence_id: int,
    fps: int,
) -> List[pd.DataFrame]:
    """
    Get the eye tracking data grouped by single sequence experiment.

    Args:
        experiment_id (int): The experiment ID.
        session_id (int): The session ID.
        participant_ids (int | None): The participant IDs.
        sequence_id (int): The sequence ID.
        fps (int): The frames per second.

    Raises:
        ValueError: If no data is found for the provided ids.

    Returns:
        List[pd.DataFrame]: The eye tracking data grouped by single sequence experiment.
    """
    print(
        f"⌛ Getting eye tracking data of {"all" if participant_ids is None else len(participant_ids)} participant(s) for experiment {experiment_id}, session {session_id}, and sequence {sequence_id}..."
    )

    # Get data and group by single sequence experiment
    data = get_eye_tracking_data(
        experiment_id=experiment_id,
        session_id=session_id,
        participant_ids=participant_ids,
        sequence_ids=[sequence_id],
    )
    data = data.groupby(["ExperimentId", "SessionId", "ParticipantId", "SequenceId"])
    groups = [data.get_group(group) for group in data.groups]

    if not groups:
        raise ValueError(
            f"❌ No data found for experiment {experiment_id}, session {session_id}, and participant(s) {participant_ids}."
        )

    # Sort each sequence by timestamp and compute time difference between consecutive frames
    for i, group in enumerate(groups):
        group = group.copy()
        group = group.sort_values("Timestamp")
        group["TimeDiff"] = group["Timestamp"].diff().fillna(0)
        group["FrameDiff"] = group["TimeDiff"] / N_HNANOSECONDS_IN_SECONDS * fps
        group["FrameNumber"] = group["FrameDiff"].cumsum().astype(int)
        group.drop(columns=["TimeDiff", "FrameDiff"], inplace=True)
        groups[i] = group

    print("✅ Eye tracking data loaded.")

    return groups


def __get_background(
    experiment_id: int,
    session_id: int,
    sequence_id: int,
    frame_width: int,
    frame_height: int,
) -> Tuple[np.ndarray | List[np.ndarray], float | None]:
    """
    Get the background for the given experiment, session, and sequence.

    Args:
        experiment_id (int): The experiment ID.
        session_id (int): The session ID.
        sequence_id (int): The sequence ID.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
    """
    print(
        f"⌛ Getting background for experiment {experiment_id}, session {session_id}, and sequence {sequence_id}..."
    )

    background_file_path = f"{SETS_PATH}/experiment{experiment_id}"

    # First session of the first experiment has images as background
    if experiment_id == 1 and session_id == 1:
        background_file_path += f"/images/scene{sequence_id}.png"
        background = cv2.imread(background_file_path)
        background = cv2.resize(background, (frame_width, frame_height))
        background_fps = None
    # Other sessions have videos as background
    else:
        if experiment_id == 1 and session_id == 2:
            background_file_path += f"/videos/scene{sequence_id}.mp4"
        elif experiment_id == 2 and session_id == 1:
            background_file_path += f"/clear/scene{sequence_id}.mp4"
        elif experiment_id == 2 and session_id == 2:
            background_file_path += f"/overcast/scene{sequence_id}.mp4"
        video_capture = cv2.VideoCapture(background_file_path)
        background_fps = video_capture.get(cv2.CAP_PROP_FPS)
        background = []
        while True:
            ret, frame = video_capture.read()

            if not ret:
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            background.append(frame)
        video_capture.release()

    print("✅ Background loaded.")

    return background, background_fps

def __get_background_frame(
        background: np.ndarray | List[np.ndarray],
        curr_frame: int,
        background_fps: float | None,
        fps: int,
    ) -> np.ndarray:
    """
    Get the background frame for the given frame number.
    
    Args:
        background (np.ndarray | List[np.ndarray]): The background image or video.
        curr_frame (int): The current frame number.
        background_fps (float | None): The background frames per second.
        fps (int): The frames per second.
    """
    if isinstance(background, np.ndarray):
        frame = background.copy()
    else:
        curr_background_frame = int(curr_frame * background_fps / fps)
        darken_ratio = 1 if curr_background_frame < len(background) else END_BACKGROUND_DARK_RATIO
        curr_background_frame = min(curr_background_frame, len(background) - 1)
        frame = background[curr_background_frame].copy()
        frame = (frame * darken_ratio).astype(np.uint8)

    return frame

def __update_coordinates_buffers(
        groups: List[pd.DataFrame],
        next_frames: List[int],
        curr_frame: int,
        coordinates_buffers: List[CoordinatesBuffer],
        max_frame: int,
    ) -> List[CoordinatesBuffer]:
    """
    Update the coordinates buffers with the current gaze coordinates.
    
    Args:
        groups (List[pd.DataFrame]): The eye tracking data grouped by single sequence experiment.
        next_frames (List[int]): The next frame for each group.
        curr_frame (int): The current frame number.
        coordinates_buffers (List[CoordinatesBuffer]): The coordinates buffers.
        max_frame (int): The maximum frame number.

    Returns:
        List[CoordinatesBuffer]: The updated coordinates buffers.
    """
    for i, group in enumerate(groups):
        next_frame = next_frames[i]
        if curr_frame == next_frame:
            # Update current gaze coordinates
            curr_group_coordinates = group[group["FrameNumber"] == curr_frame][
                ["GazeX", "GazeY"]
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

    return coordinates_buffers

def __draw_gaze_sequence(
    coordinates_buffers: List[CoordinatesBuffer],
    frame: np.ndarray,
    frame_width: int,
    frame_height: int,
    circle_radius: int,
    line_thickness: int,
) -> np.ndarray:
    """
    Draw the gaze sequence on the frame.

    Args:
        coordinates_buffers (List[CoordinatesBuffer]): The coordinates buffers.
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
        for ((x1, y1), (x2, y2)) in zip(coordinate_buffer, coordinate_buffer[1:]):
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

    overlay_mask = overlay.sum(axis=2) > 0
    frame[overlay_mask] = overlay[overlay_mask]

    return frame

def __draw_information(
    frame: np.ndarray,
    curr_frame: int,
    max_frame: int,
    experiment_id: int,
    session_id: int,
    sequence_id: int,
    frame_width: int,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 0.5,
    margin: int = 20,
) -> np.ndarray:
    """
    Draw the information on the frame.

    Args:
        frame (np.ndarray): The frame.
        curr_frame (int): The current frame number.
        max_frame (int): The maximum frame number.
        experiment_id (int): The experiment ID.
        session_id (int): The session ID.
        sequence_id (int): The sequence ID.
        frame_width (int): The frame width.
        font (int, optional): The font. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): The font scale. Defaults to 0.5.
        margin (int, optional): The margin. Defaults to 20.

    Returns:
        np.ndarray: The frame with the information drawn.
    """
    # Draw experiment, session, and sequence IDs
    id_text = f"Experiment {experiment_id}, Session {session_id}, Sequence {sequence_id}"
    id_text_x = margin
    id_text_y = margin
    cv2.putText(
        frame,
        id_text,
        (id_text_x, id_text_y),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # Draw current frame number
    n_digits = len(str(max_frame))
    frame_text = f"Frame {str(curr_frame).zfill(n_digits)}/{max_frame}"
    frame_text_size = cv2.getTextSize(frame_text, font, font_scale, 1)[0]
    frame_text_x = frame_width - frame_text_size[0] - margin
    frame_text_y = margin
    cv2.putText(
        frame,
        frame_text,
        (frame_text_x, frame_text_y),
        font,
        font_scale,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    return frame

def __visualize_gaze_sequence(
    experiment_id: int,
    session_id: int,
    participant_ids: int | None,
    sequence_id: int,
    output_file_path: str,
    frame_width: int,
    frame_height: int,
    fps: int,
    circle_radius: int,
    line_thickness: int,
    trail_length: int
) -> None:
    """
    Visualize the gaze sequence for the given experiment, session, participant(s), and sequence.
    
    Args:
        experiment_id (int): The experiment ID.
        session_id (int): The session ID.
        participant_ids (int | None): The participant IDs.
        sequence_id (int): The sequence ID.
        output_file_path (str): The output file path.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        fps (int): The frames per second.
        circle_radius (int): The circle radius.
        line_thickness (int): The line thickness.
        trail_length (int): The trail length.
    """
    # Get eye tracking data
    groups = __get_grouped_eye_tracking_data(
        experiment_id=experiment_id,
        session_id=session_id,
        participant_ids=participant_ids,
        sequence_id=sequence_id,
        fps=fps,
    )

    # Get background image or video
    background, background_fps = __get_background(
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
    coordinates_buffers = [CoordinatesBuffer(max_length=trail_length) for _ in groups]
    next_frames = [group["FrameNumber"].iloc[0] for group in groups]
    max_frame = max([group["FrameNumber"].max() for group in groups])
    bar = tqdm(total=max_frame, desc="⌛ Generating gaze video...", unit="frames")
    while curr_frame < max_frame:

        # Get current background frame
        frame = __get_background_frame(
            background=background,
            curr_frame=curr_frame,
            background_fps=background_fps,
            fps=fps,
        )

        # Update current gaze coordinates
        coordinates_buffers = __update_coordinates_buffers(
            groups=groups,
            next_frames=next_frames,
            curr_frame=curr_frame,
            coordinates_buffers=coordinates_buffers,
            max_frame=max_frame,
        )

        # Draw gaze coordinates and information on frame
        frame = __draw_gaze_sequence(
            coordinates_buffers=coordinates_buffers,
            frame=frame,
            frame_width=frame_width,
            frame_height=frame_height,
            circle_radius=circle_radius,
            line_thickness=line_thickness,
        )

        frame = __draw_information(
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

def parse_arguments() -> argparse.Namespace:
    """
    Parse the command line arguments.
    
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
        default=f"{GENERATED_PATH}/gaze_sequence.mp4",
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
        "--speed",
        "-s",
        type=int,
        default=1,
        help="The speed.",
    )

    return parser.parse_args()

def main() -> None:
    """
    Main function for visualizing gaze sequence.
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

    __visualize_gaze_sequence(
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
    )


if __name__ == "__main__":
    main()
