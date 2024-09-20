import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import pandas as pd
from typing import List

from src.config import (
    PROCESSED_EYE_TRACKING_DATA_PATH,
    PROCESSED_EYE_TRACKING_FILE_NAME,
)


def get_eye_tracking_data(
    experiment_id: int | None = None,
    session_id: int | None = None,
    participant_ids: List[int] | None = None,
    sequence_ids: List[int] | None = None,
    interpolated: bool = False,
) -> pd.DataFrame:
    """
    Get eye tracking data for a specific experiment, session, and participant. If no id is provided, all data is returned.

    Args:
        experiment_id (int | None, optional): The experiment id. Defaults to None.
        session_id (int | None, optional): The session id. Defaults to None.
        participant_ids (List[int] | None, optional): The participant id. Defaults to None.
        interpolated (bool, optional): Whether to return the interpolated data. Defaults to False.

    Raises:
        ValueError: If the experiment id is not 1 or 2.
        ValueError: If the session id is not 1 or 2.
        ValueError: If no data is found for the provided ids.
    """
    if experiment_id is not None and experiment_id not in [1, 2]:
        raise ValueError(f"❌ Invalid experiment id: {experiment_id}, must be 1 or 2.")
    if session_id is not None and session_id not in [1, 2]:
        raise ValueError(f"❌ Invalid session id: {session_id}, must be 1 or 2.")

    data_file_path = f"{PROCESSED_EYE_TRACKING_DATA_PATH}/{'interpolated_' if interpolated else ''}{PROCESSED_EYE_TRACKING_FILE_NAME}"
    data = pd.read_csv(data_file_path)

    if experiment_id is not None:
        data = data[data["ExperimentId"] == experiment_id]
    if session_id is not None:
        data = data[data["SessionId"] == session_id]
    if participant_ids is not None:
        data = data[data["ParticipantId"].isin(participant_ids)]
    if sequence_ids is not None:
        data = data[data["SequenceId"].isin(sequence_ids)]

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
        ["ExperimentId", "SessionId", "ParticipantId", "SequenceId"]
    )
    groups = [grouped_data.get_group(group) for group in grouped_data.groups]

    for i, group in enumerate(groups):
        group = group.copy()
        group = group.sort_values("Timestamp_ns")
        group["TimeDiff_ns"] = group["Timestamp_ns"].diff().fillna(0)
        group["TimeSinceStart_ns"] = group["TimeDiff_ns"].cumsum()
        group = group.drop(columns=["TimeDiff_ns"])
        groups[i] = group
    data = pd.concat(groups)

    return data
