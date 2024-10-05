import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple

from src.utils.eye_tracking_data import (
    get_eye_tracking_data,
    with_time_since_start_column,
    with_time_since_start_end_column,
)
from src.utils.saliency import get_kde_density, get_saliency_map
from src.config import SETS_PATH

N_NANOSECONDS_IN_SECOND = 1e9  # Number of hundred nanoseconds in a second
END_BACKGROUND_DARK_RATIO = 0.5
SALIENCY_COLORMAP = cv2.COLORMAP_HOT


def get_grouped_processed_data(
    experiment_ids: List[int] | None,
    session_ids: List[int] | None,
    participant_ids: List[int] | None,
    sequence_ids: List[int] | None,
    set_ids: List[int] | None,
    fps: int,
    interpolated: bool,
) -> List[pd.DataFrame]:
    """
    Get the eye tracking data grouped by single sequence experiment.

    Args:
        experiment_id (List[int] | None): The experiment ID.
        session_id (List[int] | None): The session ID.
        participant_ids (List[int] | None): The participant IDs.
        sequence_id (List[int] | None): The sequence ID.
        fps (int): The frames per second.
        interpolated (bool): Whether to return the interpolated data.

    Raises:
        ValueError: If no data is found for the provided ids.

    Returns:
        List[pd.DataFrame]: The eye tracking data grouped by single sequence experiment.
    """
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
    print(
        f"⌛ Getting eye tracking data of {participants_str} for {', '.join([experiment_str, session_str, sequence_str, set_str])}..."
    )

    # Get data and group by single sequence experiment
    data = get_eye_tracking_data(
        experiment_ids=experiment_ids,
        session_ids=session_ids,
        participant_ids=participant_ids,
        sequence_ids=sequence_ids,
        set_ids=set_ids,
        interpolated=interpolated,
    )
    data = with_time_since_start_column(data)
    data = data.groupby(
        ["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"]
    )
    groups = [data.get_group(group) for group in data.groups]

    if not groups:
        raise ValueError(
            f"❌ No data found for experiment(s) {experiment_ids}, session(s) {session_ids}, participant(s) {participant_ids}, sequence(s) {sequence_ids}, and set(s) {set_ids}."
        )

    # Add frame number
    for i, group in enumerate(groups):
        group = group.copy()
        group["FrameNumber"] = (
            group["TimeSinceStart_ns"] / N_NANOSECONDS_IN_SECOND * fps
        )
        group["FrameNumber"] = group["FrameNumber"].astype(int)
        groups[i] = group
    print("✅ Processed data loaded.")

    return groups


def get_grouped_fixation_data(
    experiment_ids: List[int] | None,
    session_ids: List[int] | None,
    participant_ids: List[int] | None,
    sequence_ids: List[int] | None,
    set_ids: List[int] | None,
    fps: int,
    processed_groups: List[pd.DataFrame],
) -> List[pd.DataFrame]:
    """
    Get the fixation data grouped by single sequence experiment.

    Args:
        experiment_id (int): The experiment ID.
        session_id (int): The session ID.
        participant_ids (List[int] | None): The participant IDs.
        sequence_id (int): The sequence ID.
        set_ids (List[int] | None): The set IDs.
        fps (int): The frames per second.
        processed_groups (List[pd.DataFrame]): The processed gaze data grouped by single sequence experiment. Used to calibrate the start and end frame numbers since the first fixation entry does not necessarily start at the beginning of the sequence.

    Raises:
        ValueError: If no data is found for the provided ids.

    Returns:
        List[pd.DataFrame]: The fixation data grouped by single sequence experiment.
    """
    # Get data and group by single sequence experiment
    data = get_eye_tracking_data(
        experiment_ids=experiment_ids,
        session_ids=session_ids,
        participant_ids=participant_ids,
        sequence_ids=sequence_ids,
        set_ids=set_ids,
        fixation=True,
    )
    data = with_time_since_start_end_column(
        data=data, processed_groups=processed_groups
    )
    data = data.groupby(
        ["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"]
    )
    groups = [data.get_group(group) for group in data.groups]

    if not groups:
        raise ValueError(
            f"❌ No data found for experiment(s) {experiment_ids}, session(s) {session_ids}, participant(s) {participant_ids}, sequence(s) {sequence_ids}, and set(s) {set_ids}."
        )

    # Add start and end frame numbers
    for i, group in enumerate(groups):
        group = group.copy()
        group["StartFrameNumber"] = (
            group["StartTimeSinceStart_ns"] / N_NANOSECONDS_IN_SECOND * fps
        )
        group["StartFrameNumber"] = group["StartFrameNumber"].astype(int)
        group["EndFrameNumber"] = (
            group["EndTimeSinceStart_ns"] / N_NANOSECONDS_IN_SECOND * fps
        )
        group["EndFrameNumber"] = group["EndFrameNumber"].astype(int)
        groups[i] = group
    print("✅ Fixation data loaded.")

    return groups


def get_background(
    experiment_id: int,
    sequence_id: int,
    set_id: int,
    frame_width: int,
    frame_height: int,
    only_first_frame: bool = False,
) -> Tuple[np.ndarray | List[np.ndarray], float | None]:
    """
    Get background for the given experiment, session, and sequence.

    Args:
        experiment_id (int): The experiment ID.
        sequence_id (int): The sequence ID.
        set_id (int): The set ID.
        frame_width (int): The frame width.
        frame_height (int): The frame height.
        only_first_frame (bool, optional): Whether to return only the first frame of the video. Defaults to False.
    """
    print(
        f"⌛ Getting background for experiment {experiment_id}, sequence {sequence_id} and set {set_id}..."
    )

    background_file_path = f"{SETS_PATH}/experiment{experiment_id}"

    # First session of the first experiment has images as background
    if experiment_id == 1 and set_id == 1:
        background_file_path += f"/images/scene{sequence_id}.png"
        background = cv2.imread(background_file_path)
        background = cv2.resize(background, (frame_width, frame_height))
        background_fps = None
    # Other sessions have videos as background
    else:
        # Get video
        if experiment_id == 1 and set_id == 0:
            background_file_path += f"/videos/scene{sequence_id}.mp4"
        elif experiment_id == 2 and set_id == 0:
            background_file_path += f"/clear/scene{sequence_id}.mp4"
        elif experiment_id == 2 and set_id == 1:
            background_file_path += f"/overcast/scene{sequence_id}.mp4"
        video_capture = cv2.VideoCapture(background_file_path)
        background_fps = video_capture.get(cv2.CAP_PROP_FPS)
        background = []
        while True:
            ret, frame = video_capture.read()

            # Break if only the first frame is requested and it is already loaded
            # or if the video is over
            if not ret or (only_first_frame and background):
                break

            frame = cv2.resize(frame, (frame_width, frame_height))
            background.append(frame)
        video_capture.release()

        # Take only the first frame if requested
        if only_first_frame:
            background = background[0]

    print("✅ Background loaded.")

    return background, background_fps


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
        darken_ratio = (
            1 if curr_background_frame < len(background) else END_BACKGROUND_DARK_RATIO
        )
        curr_background_frame = min(curr_background_frame, len(background) - 1)
        frame = background[curr_background_frame].copy()
        frame = (frame * darken_ratio).astype(np.uint8)

    return frame


def draw_saliency(
    coordinates: List[Tuple[float, float]],
    frame: np.ndarray,
    frame_width: int,
    frame_height: int,
    saliency_width: int,
    saliency_height: int,
    kde_bandwidth: float,
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

    frame = frame.copy()

    # Rescale gaze coordinates to frame size
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
        kde_bandwidth=kde_bandwidth,
        apply_exponential=True,
    )
    saliency_map = get_saliency_map(
        kde_density=kde_density,
        frame_width=frame_width,
        frame_height=frame_height,
        saliency_colormap=SALIENCY_COLORMAP,
    )
    frame = cv2.addWeighted(frame, 0.5, saliency_map, 0.5, 0)

    return frame


def draw_information(
    frame: np.ndarray,
    curr_frame: int,
    max_frame: int,
    experiment_ids: List[int] | None,
    session_ids: List[int] | None,
    sequence_ids: List[int] | None,
    set_ids: List[int] | None,
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
        experiment_ids (List[int] | None): The experiment IDs.
        session_ids (List[int] | None): The session IDs.
        sequence_ids (List[int] | None): The sequence IDs.
        set_ids (List[int] | None): The set IDs.
        frame_width (int): The frame width.
        font (int, optional): The font. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): The font scale. Defaults to 0.5.
        margin (int, optional): The margin. Defaults to 20.

    Returns:
        np.ndarray: The frame with the information drawn.
    """
    # Draw experiment, session, and sequence IDs
    id_text = ""
    id_text += (
        f"Experiment(s) {', '.join(map(str, experiment_ids))} "
        if experiment_ids
        else ""
    )
    id_text += f"Session(s) {', '.join(map(str, session_ids))} " if session_ids else ""
    id_text += (
        f"Sequence(s) {', '.join(map(str, sequence_ids))} " if sequence_ids else ""
    )
    id_text += f"Set(s) {', '.join(map(str, set_ids))} " if set_ids else ""

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
