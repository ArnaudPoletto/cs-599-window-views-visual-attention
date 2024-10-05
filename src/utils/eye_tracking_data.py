import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import numpy as np
import pandas as pd
from typing import List

from src.config import (
    EYE_TRACKING_PROCESSED_PATH,
    PROCESSED_EYE_TRACKING_FILE_NAME,
)

N_NANOSECONDS_IN_SECOND = 1e9  # Number of nanoseconds in a second


def get_eye_tracking_data(
    experiment_ids: List[int] | None = None,
    session_ids: List[int] | None = None,
    participant_ids: List[int] | None = None,
    sequence_ids: List[int] | None = None,
    set_ids: List[int] | None = None,
    interpolated: bool = False,
    fixation: bool = False,
) -> pd.DataFrame:
    """
    Get eye tracking data for a specific experiment, session, and participant. If no id is provided, all data is returned.

    Args:
        experiment_ids (List[int] | None, optional): The experiment id. Defaults to None.
        session_ids (List[int] | None, optional): The session id. Defaults to None.
        participant_ids (List[int] | None, optional): The participant id. Defaults to None.
        sequence_ids (List[int] | None, optional): The sequence id. Defaults to None.
        set_ids (List[int] | None, optional): The set id. Defaults to None.
        interpolated (bool, optional): Whether to return the interpolated data. Defaults to False.
        fixation (bool, optional): Whether to return the fixation data. Defaults to False.

    Raises:
        ValueError: If the experiment ids are not 1 or 2.
        ValueError: If the session ids are not 1 or 2.
        ValueError: If both interpolated and fixation data are requested.
        ValueError: If no data is found for the provided ids.
    """
    if experiment_ids is not None and set(experiment_ids) - {1, 2}:
        raise ValueError(f"❌ Invalid experiment ids: {experiment_ids}, must contain 1 or 2.")
    if session_ids is not None and set(session_ids) - {1, 2}:
        raise ValueError(f"❌ Invalid session ids: {session_ids}, must contain 1 or 2.")
    if set_ids is not None and set(set_ids) - {0, 1}:
        raise ValueError(f"❌ Invalid set ids: {set_ids}, must contain 0 or 1.")
    if interpolated and fixation:
        raise ValueError("❌ Either interpolated or fixation data can be returned, not both.")

    data_file_prefix = "interpolated_" if interpolated else "fixation_" if fixation else ""
    data_file_path = f"{EYE_TRACKING_PROCESSED_PATH}/{data_file_prefix}{PROCESSED_EYE_TRACKING_FILE_NAME}"
    data = pd.read_csv(data_file_path)

    if experiment_ids is not None:
        data = data[data["ExperimentId"].isin(experiment_ids)]
    if session_ids is not None:
        data = data[data["SessionId"].isin(session_ids)]
    if participant_ids is not None:
        data = data[data["ParticipantId"].isin(participant_ids)]
    if sequence_ids is not None:
        data = data[data["SequenceId"].isin(sequence_ids)]
    if set_ids is not None:
        data = data[data["SetId"].isin(set_ids)]

    return data

def with_media_type_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column with the media type of the stimulus.

    Args:
        data (pd.DataFrame): The eye tracking data.

    Returns:
        pd.DataFrame: The eye tracking data with the media type column.
    """
    if "MediaType" in data.columns:
        print(" ⚠️  MediaType column already exists in the data, skipping...")
        return data

    if data.empty:
        print(" ⚠️  No data provided, skipping...")
        data["MediaType"] = ""
        return data

    is_image = (data["ExperimentId"] == 1) & (data["SetId"] == 1)
    data["MediaType"] = is_image.map({True: "Image", False: "Video"})

    return data

def with_time_since_start_column(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add a column with the time since the start of the experiment in hundredths of a nanosecond.

    Args:
        data (pd.DataFrame): The eye tracking data.

    Returns:
        pd.DataFrame: The eye tracking data with the time difference column.
    """
    if "TimeSinceStart_ns" in data.columns:
        print(" ⚠️  TimeSinceStart_ns column already exists in the data, skipping...")
        return data

    if data.empty:
        print(" ⚠️  No data provided, skipping...")
        data["TimeSinceStart_ns"] = 0
        return data

    grouped_data = data.groupby(
        ["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"]
    )
    groups = [grouped_data.get_group(group) for group in grouped_data.groups]

    for i, group in enumerate(groups):
        timestamp_column = "Timestamp_ns" if "Timestamp_ns" in group.columns else "StartTimestamp_ns"
        start_time = group[timestamp_column].min()
        group = group.copy()
        group["TimeSinceStart_ns"] = group[timestamp_column] - start_time
        groups[i] = group
    data = pd.concat(groups)

    return data


def with_time_since_start_end_column(
        data: pd.DataFrame,
        processed_groups: List[pd.DataFrame] | None = None,
        ) -> pd.DataFrame:
    """
    Add columns with the time since the start and end of the experiment in hundredths of a nanosecond.
    
    Args:
        data (pd.DataFrame): The eye tracking data.
        processed_groups (List[pd.DataFrame] | None, optional): The processed gaze data grouped by single sequence experiment. Used to calibrate the start and end frame numbers since the first fixation entry does not necessarily start at the beginning of the sequence. Defaults to None.

    Returns:
        pd.DataFrame: The eye tracking data with the time difference columns.
    """
    if "StartTimeSinceStart_ns" in data.columns and "EndTimeSinceStart_ns" in data.columns:
        print(" ⚠️  StartTimeSinceStart_ns and EndTimeSinceStart_ns columns already exist in the data, skipping...")
        return data
    
    if data.empty:
        print(" ⚠️  No data provided, skipping...")
        data["StartTimeSinceStart_ns"] = 0
        data["EndTimeSinceStart_ns"] = 0
        return data
    
    grouped_data = data.groupby(
        ["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"]
    )
    groups = [grouped_data.get_group(group) for group in grouped_data.groups]

    for i, group in enumerate(groups):
        group = group.copy()

        # Get start time either by getting the minimum start timestampof the data itself or from the corresponding group
        if processed_groups is None:
            start_time = group["StartTimestamp_ns"].min()
        else:
            # get Timestamp_ns min from the corresponding group
            corresponding_group = next((
                g for g in processed_groups if 
                g["ExperimentId"].iloc[0] == group["ExperimentId"].iloc[0] and
                g["SessionId"].iloc[0] == group["SessionId"].iloc[0] and
                g["ParticipantId"].iloc[0] == group["ParticipantId"].iloc[0] and
                g["SequenceId"].iloc[0] == group["SequenceId"].iloc[0] and
                g["SetId"].iloc[0] == group["SetId"].iloc[0]
            ), None)
            if corresponding_group is None:
                print(" ⚠️  No corresponding group found, getting the minimum start timestamp of the data itself.")
                start_time = group["StartTimestamp_ns"].min()
            else:
                start_time = corresponding_group["Timestamp_ns"].min()

        group["StartTimeSinceStart_ns"] = group["StartTimestamp_ns"] - start_time
        group["EndTimeSinceStart_ns"] = group["EndTimestamp_ns"] - start_time
        groups[i] = group
    data = pd.concat(groups)

    return data

def with_time_since_last_column(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a column with the time since the previous gaze point in nanoseconds.

    Args:
        data (pd.DataFrame): The eye tracking data.

    Returns:
        pd.DataFrame: The eye tracking data with the time since last gaze point column.
    """
    data = data.groupby(["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"])
    groups = [data.get_group(group) for group in data.groups]

    for i, group in enumerate(groups):
        timestamp_column = "Timestamp_ns" if "Timestamp_ns" in group.columns else "StartTimestamp_ns"
        group = group.copy()
        group = group.sort_values(timestamp_column)
        time_since_last = group[timestamp_column].diff().fillna(0)
        group["TimeSinceLast_ns"] = time_since_last
        groups[i] = group
    data = pd.concat(groups)

    return data


def with_distance_since_last_column(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a column with the distance since the previous gaze point in pixels.

    Args:
        data (pd.DataFrame): The eye tracking data.

    Returns:
        pd.DataFrame: The eye tracking data with the distance since last gaze point column.
    """
    data = data.groupby(["ExperimentId", "SessionId", "ParticipantId", "SequenceId", "SetId"])
    groups = [data.get_group(group) for group in data.groups]

    for i, group in enumerate(groups):
        timestamp_column = "Timestamp_ns" if "Timestamp_ns" in group.columns else "StartTimestamp_ns"
        group = group.copy()
        group = group.sort_values(timestamp_column)
        gaze_x_diff = group["X_px"].diff().fillna(0)
        gaze_y_diff = group["Y_px"].diff().fillna(0)
        distance_since_last = np.sqrt(gaze_x_diff**2 + gaze_y_diff**2)
        group["DistanceSinceLast_px"] = distance_since_last
        groups[i] = group
    data = pd.concat(groups)

    return data


def with_speed_since_last_column(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a column with the speed between the current and the previous gaze point in pixels.

    Args:
        data (pd.DataFrame): The eye tracking data.

    Returns:
        pd.DataFrame: The eye tracking data with the speed since last gaze point column.
    """
    # Group the data by experiment, session, participant, and sequence
    data = data.copy()
    data = with_time_since_last_column(data)
    data = with_distance_since_last_column(data)
    data["SpeedSinceLast_pxs"] = data["DistanceSinceLast_px"] / (
        data["TimeSinceLast_ns"] / N_NANOSECONDS_IN_SECOND
    )

    return data
