import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import pandas as pd
from typing import List

from src.config import PROCESSED_EYE_TRACKING_DATA_PATH


def get_eye_tracking_data(
    experiment_id: int | None = None,
    session_id: int | None = None,
    participant_ids: List[int] | None = None,
    sequence_ids: List[int] | None = None,
) -> pd.DataFrame:
    """
    Get eye tracking data for a specific experiment, session, and participant. If no id is provided, all data is returned.

    Args:
        experiment_id (int | None, optional): The experiment id. Defaults to None.
        session_id (int | None, optional): The session id. Defaults to None.
        participant_ids (List[int] | None, optional): The participant id. Defaults to None.
    """
    if experiment_id is not None and experiment_id not in [1, 2]:
        raise ValueError(f"❌ Invalid experiment id: {experiment_id}, must be 1 or 2.")
    if session_id is not None and session_id not in [1, 2]:
        raise ValueError(f"❌ Invalid session id: {session_id}, must be 1 or 2.")

    # Get files to load
    eye_tracking_data_path = PROCESSED_EYE_TRACKING_DATA_PATH
    if experiment_id is not None:
        eye_tracking_data_path += f"/experiment{experiment_id}"
    if session_id is not None:
        eye_tracking_data_path += f"/session{session_id}"

    file_paths = Path(eye_tracking_data_path).rglob(f"participant*_*.csv")
    if participant_ids is not None:
        participant_ids_str = [
            f"{participant_id:02d}" for participant_id in participant_ids
        ]
        file_paths = [
            p
            for p in file_paths
            if any(p.stem.startswith(f"participant{pid}_") for pid in participant_ids_str)
        ]

    # Load data
    data = None
    for file_path in file_paths:
        file_data = pd.read_csv(file_path)

        # Filter by sequence ids
        if sequence_ids is not None:
            file_data = file_data[file_data["SequenceId"].isin(sequence_ids)]

        if data is None:
            data = file_data
        else:
            data = pd.concat([data, file_data])

    return data
