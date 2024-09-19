import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple

from src.utils.eye_tracking_data import get_eye_tracking_data
from src.config import SETS_PATH

N_HNANOSECONDS_IN_SECONDS = 1e7  # Number of hundred nanoseconds in a second
END_BACKGROUND_DARK_RATIO = 0.5

def get_grouped_eye_tracking_data(
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


def get_background(
    experiment_id: int,
    session_id: int,
    sequence_id: int,
    frame_width: int,
    frame_height: int,
) -> Tuple[np.ndarray | List[np.ndarray], float | None]:
    """
    Get background for the given experiment, session, and sequence.

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


def get_max_frame(
    groups: List[pd.DataFrame],
    background: np.ndarray | List[np.ndarray],
    end_at_video_end: bool,
) -> int:
    """
    Get the maximum frame number.
    
    Args:
        groups (List[pd.DataFrame]): The grouped eye tracking data.
        background (np.ndarray | List[np.ndarray]): The background image or video.
        
    Returns:
        int: The maximum frame number.
    """
    if end_at_video_end and isinstance(background, List):
        max_frame = len(background)
    else:
        max_frame = max([group["FrameNumber"].max() for group in groups])

    return max_frame


def get_background_frame(
        background: np.ndarray | List[np.ndarray],
        curr_frame: int,
        background_fps: float | None,
        fps: int,
    ) -> np.ndarray:
    """
    Get background frame for the given frame number.
    
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


def draw_information(
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
    Draw information on the frame.

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
    id_text = (
        f"Experiment {experiment_id}, Session {session_id}, Sequence {sequence_id}"
    )
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
