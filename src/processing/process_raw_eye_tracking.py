import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import pandas as pd
from tqdm import tqdm

from src.utils.eye_tracking_data import with_time_since_start_column
from src.config import (
    RAW_EYE_TRACKING_DATA_PATH,
    RAW_EYE_TRACKING_FRAME_WIDTH,
    RAW_EYE_TRACKING_FRAME_HEIGHT,
    PROCESSED_EYE_TRACKING_DATA_PATH,
    PROCESSED_EYE_TRACKING_FILE_NAME,
)

OUTLIER_VALUES = (3000, 1500)
MAX_TIME_SINCE_START_SECONDS = 120
N_HNANOSECONDS_IN_NANOSECOND = 100  # Number of hundred nanoseconds in a nanosecond
N_NANOSECONDS_IN_SECOND = 1e9  # Number of nanoseconds in a second

# Offset between Windows FileTime epoch (1601-01-01) and Unix epoch (1970-01-01) in hundreds of nanoseconds
WINDOWS_TO_UNIX_EPOCH_OFFSET_NS = 116444736000000000


def get_raw_data() -> pd.DataFrame:
    """
    Get raw eye tracking data.

    Returns:
        pd.DataFrame: The raw eye tracking data.
    """
    # Get valid source file paths
    file_paths = Path(RAW_EYE_TRACKING_DATA_PATH).rglob(
        "Exp[12]_[12][0-9][0-9][12]_*.csv"
    )
    file_paths = [file_path.resolve().as_posix() for file_path in file_paths]
    n_files = len(file_paths)

    # Check if all files were found
    all_file_paths = Path(RAW_EYE_TRACKING_DATA_PATH).rglob("*.csv")
    all_file_paths = [file_path.resolve().as_posix() for file_path in all_file_paths]
    ignored_files = set(all_file_paths) - set(file_paths)
    if len(ignored_files) > 0:
        print(
            f"➡️  Found {n_files} raw eye tracking data files, ignoring the following {len(ignored_files)} file(s):"
        )
        for ignored_file in ignored_files:
            print(f"\t - {ignored_file}")
    else:
        print(f"➡️  Found {n_files} raw eye tracking data files.")

    # Read raw eye tracking data
    raw_data_list = []
    for file_path in tqdm(
        file_paths, total=n_files, desc="⌛ Reading raw eye tracking data"
    ):
        raw_file_data = pd.read_csv(file_path, sep=";")
        raw_data_list.append(raw_file_data)

    raw_data = pd.concat(raw_data_list, axis=0, ignore_index=True)

    return raw_data


def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Process the raw eye tracking data.

    Args:
        data (pd.DataFrame): The raw eye tracking data.

    Returns:
        pd.DataFrame: The processed eye tracking data.
    """
    print("⌛ Processing raw eye tracking data.")
    data = data.copy()

    # Delete entries with NaN values
    data = data.dropna()

    # Delete false center gaze points
    data = data[
        (data["GazeX"] != OUTLIER_VALUES[0]) & (data["GazeY"] != OUTLIER_VALUES[1])
    ]

    # Add gaze screen coordinates column and rename gaze pixel coordinates column
    data["GazeX_sc"] = data["GazeX"] / RAW_EYE_TRACKING_FRAME_WIDTH
    data["GazeY_sc"] = data["GazeY"] / RAW_EYE_TRACKING_FRAME_HEIGHT
    data = data.rename(columns={"GazeX": "GazeX_px", "GazeY": "GazeY_px"})

    # Get experiment, session and participant ids
    data["ExperimentId"] = data["Id"] // 1000  # Is the thousands digit
    data["SessionId"] = data["Id"] % 10  # Is the unit digit
    data["ParticipantId"] = (data["Id"] % 1000) // 10  # Is the hundreds and tens digit

    # Delete entries with invalid ids
    data = data[
        ((data["ExperimentId"] == 1) | (data["ExperimentId"] == 2))
        & ((data["SessionId"] == 1) | (data["SessionId"] == 2))
    ]

    # Change timestamp unit to nanoseconds
    data["Timestamp"] = data["Timestamp"].astype("int64")
    data["Timestamp_ns"] = data["Timestamp"] * N_HNANOSECONDS_IN_NANOSECOND

    # Delete entries recorded after a long time
    data = with_time_since_start_column(data)
    data = data[
        data["TimeSinceStart_ns"]
        <= MAX_TIME_SINCE_START_SECONDS * N_NANOSECONDS_IN_SECOND
    ]
    data = data.drop(columns=["TimeSinceStart_ns"])

    # Delete vector gaze information and id
    data = data.drop(
        columns=[
            "VectorGazeX",
            "VectorGazeY",
            "VectorGazeZ",
            "Id",
            "SequenceSet",
            "Timestamp",
        ]
    )

    # Convert types
    data = data.astype(
        {
            "ExperimentId": "int",
            "SessionId": "int",
            "ParticipantId": "int",
            "SequenceId": "int",
            "GazeX_sc": "float32",
            "GazeY_sc": "float32",
            "GazeX_px": "float32",
            "GazeY_px": "float32",
            "Timestamp_ns": "int64",
        }
    )

    # Reorder columns
    data = data[
        [
            "ExperimentId",
            "SessionId",
            "ParticipantId",
            "SequenceId",
            "GazeX_sc",
            "GazeY_sc",
            "GazeX_px",
            "GazeY_px",
            "Timestamp_ns",
        ]
    ]

    print("✅ Eye tracking data processed.")

    return data


def get_interpolated_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    # The conversion does not give good date, but the time unit is correct
    data["DateTime"] = pd.to_datetime(data["Timestamp_ns"], unit="ns")
    data = data.groupby(["ExperimentId", "SessionId", "ParticipantId", "SequenceId"])
    groups = [data.get_group(x) for x in data.groups]

    for i, group in tqdm(
        enumerate(groups), total=len(groups), desc="⌛ Interpolating eye tracking data"
    ):
        group = group.copy()
        group = group.set_index("DateTime")

        # Get group information
        experiment_id = group["ExperimentId"].iloc[0]
        session_id = group["SessionId"].iloc[0]
        participant_id = group["ParticipantId"].iloc[0]
        sequence_id = group["SequenceId"].iloc[0]

        # Resample and interpolate the data
        columns_to_interpolate = ["GazeX_sc", "GazeY_sc", "GazeX_px", "GazeY_px"]
        resampled_group = group[columns_to_interpolate].resample("50ms").mean()
        interpolated_group = resampled_group.interpolate(method="linear")
        interpolated_group = interpolated_group.reset_index()

        # Add group information
        interpolated_group["ExperimentId"] = experiment_id
        interpolated_group["SessionId"] = session_id
        interpolated_group["ParticipantId"] = participant_id
        interpolated_group["SequenceId"] = sequence_id
        interpolated_group["Timestamp_ns"] = interpolated_group["DateTime"].astype(
            "int64"
        )

        groups[i] = interpolated_group

    interpolated_data = pd.concat(groups, axis=0, ignore_index=True)

    # Reformat data
    interpolated_data = interpolated_data.drop(columns=["DateTime"])
    interpolated_data = interpolated_data.astype(
        {
            "ExperimentId": "int",
            "SessionId": "int",
            "ParticipantId": "int",
            "SequenceId": "int",
            "GazeX_sc": "float32",
            "GazeY_sc": "float32",
            "GazeX_px": "float32",
            "GazeY_px": "float32",
            "Timestamp_ns": "int64",
        }
    )

    interpolated_data = interpolated_data[
        [
            "ExperimentId",
            "SessionId",
            "ParticipantId",
            "SequenceId",
            "GazeX_sc",
            "GazeY_sc",
            "GazeX_px",
            "GazeY_px",
            "Timestamp_ns",
        ]
    ]

    return interpolated_data


def main() -> None:
    """
    Main function for processing the raw eye tracking data.
    """
    # Remove old processed eye tracking data file
    print("➡️  Removing old interpolated eye tracking data file.")
    data_file_path = (
        f"{PROCESSED_EYE_TRACKING_DATA_PATH}/{PROCESSED_EYE_TRACKING_FILE_NAME}"
    )
    interpolated_data_file_path = f"{PROCESSED_EYE_TRACKING_DATA_PATH}/interpolated_{PROCESSED_EYE_TRACKING_FILE_NAME}"
    if os.path.exists(data_file_path):
        os.remove(data_file_path)
    if os.path.exists(interpolated_data_file_path):
        os.remove(interpolated_data_file_path)

    raw_data = get_raw_data()
    processed_data = process_data(raw_data)
    interpolated_data = get_interpolated_data(processed_data)

    # Save processed eye tracking data
    os.makedirs(PROCESSED_EYE_TRACKING_DATA_PATH, exist_ok=True)
    processed_data.to_csv(data_file_path, index=False)
    os.makedirs(PROCESSED_EYE_TRACKING_DATA_PATH, exist_ok=True)
    interpolated_data.to_csv(interpolated_data_file_path, index=False)
    print(
        f"✅ Saved processed and interpolated eye tracking data to {data_file_path} and {interpolated_data_file_path}."
    )


if __name__ == "__main__":
    main()
