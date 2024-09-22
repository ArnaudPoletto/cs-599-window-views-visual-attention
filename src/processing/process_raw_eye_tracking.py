import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import argparse
import pandas as pd
from tqdm import tqdm
from typing import Any, Dict, List

from src.utils.eye_tracking_data import (
    with_time_since_start_column,
    with_distance_since_last_column,
)
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

DISPERSION_THRESHOLD_PX = 50.0
DURATION_THRESHOLD_NS = 100.0 * 1e6  # 100 ms in nanoseconds

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
    """
    Interpolate the eye tracking data.

    Args:
        data (pd.DataFrame): The eye tracking data.

    Returns:
        pd.DataFrame: The interpolated eye tracking data.
    """
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


def get_fixation_data_from_group(
    group: pd.DataFrame, dispersion_threshold_px: float, duration_threshold_ns: float
) -> List[Dict[str, Any]]:
    """
    Get fixation data from a single sequence of eye tracking data.

    Args:
        group (pd.DataFrame): The group of eye tracking data.
        dispersion_threshold_px (float): The dispersion threshold for fixation detection in pixels.
        duration_threshold_ns (float): The duration threshold for fixation detection in nanoseconds.

    Returns:
        List[Dict[str, Any]]: The fixation data.
    """
    fixation_data = []
    start_index = 0
    is_fixation = False
    for curr_index in range(len(group)):
        # Set window to cover the duration threshold
        start_timestamp = group["Timestamp_ns"].iloc[start_index]
        end_timestamp = group["Timestamp_ns"].iloc[curr_index]
        fixation_duration = end_timestamp - start_timestamp
        if fixation_duration < duration_threshold_ns:
            continue

        dispersion = (
            group["GazeX_px"].iloc[start_index:curr_index].max()
            - group["GazeX_px"].iloc[start_index:curr_index].min()
            + group["GazeY_px"].iloc[start_index:curr_index].max()
            - group["GazeY_px"].iloc[start_index:curr_index].min()
        )
        # Define the window as a fixation if it does not exceed the dispersion threshold, and increase the size of the window
        if dispersion < dispersion_threshold_px and curr_index < len(group) - 1:
            is_fixation = True
            continue
        # If the threshold is exceeded, save the fixation point and start over with next points in the time-series
        else:
            if is_fixation:
                fixation_data.append(
                    {
                        "ExperimentId": group["ExperimentId"].iloc[start_index],
                        "SessionId": group["SessionId"].iloc[start_index],
                        "ParticipantId": group["ParticipantId"].iloc[start_index],
                        "SequenceId": group["SequenceId"].iloc[start_index],
                        "FixationX_sc": group["GazeX_sc"].iloc[start_index:curr_index].mean(),
                        "FixationY_sc": group["GazeY_sc"].iloc[start_index:curr_index].mean(),
                        "FixationX_px": group["GazeX_px"].iloc[start_index:curr_index].mean(),
                        "FixationY_px": group["GazeY_px"].iloc[start_index:curr_index].mean(),
                        "StartTimestamp_ns": start_timestamp,
                        "EndTimestamp_ns": end_timestamp,
                        "Duration_ns": fixation_duration,
                    }
                )
            start_index = curr_index
            is_fixation = False

    return fixation_data


def get_fixation_data(
    data: pd.DataFrame,
    dispersion_threshold_px: float,
    duration_threshold_ns: float,
) -> pd.DataFrame:
    """
    Get fixation data from the eye tracking data.

    Args:
        data (pd.DataFrame): The eye tracking data.
        dispersion_threshold_px (float): The dispersion threshold for fixation detection in pixels.
        duration_threshold_ns (float): The duration threshold for fixation detection in nanoseconds.

    Returns:
        pd.DataFrame: The fixation data.
    """
    # Group the data by sequence
    data = data.copy()
    data = data.groupby(["ExperimentId", "SessionId", "ParticipantId", "SequenceId"])
    groups = [data.get_group(group) for group in data.groups]

    # Iterate through the data to get fixations
    fixation_data = []
    for group in tqdm(groups, total=len(groups), desc="⌛ Getting fixation data"):
        group_fixation_data = get_fixation_data_from_group(
            group, dispersion_threshold_px, duration_threshold_ns
        )
        fixation_data.extend(group_fixation_data)
    fixation_data = pd.DataFrame(fixation_data)

    # Reformat data
    fixation_data = fixation_data.astype(
        {
            "ExperimentId": "int",
            "SessionId": "int",
            "ParticipantId": "int",
            "SequenceId": "int",
            "FixationX_sc": "float32",
            "FixationY_sc": "float32",
            "FixationX_px": "float32",
            "FixationY_px": "float32",
            "StartTimestamp_ns": "int64",
            "EndTimestamp_ns": "int64",
            "Duration_ns": "int64",
        }
    )

    fixation_data = fixation_data[
        [
            "ExperimentId",
            "SessionId",
            "ParticipantId",
            "SequenceId",
            "FixationX_sc",
            "FixationY_sc",
            "FixationX_px",
            "FixationY_px",
            "StartTimestamp_ns",
            "EndTimestamp_ns",
            "Duration_ns",
        ]
    ]

    return fixation_data


def parse_arguments() -> argparse.Namespace:
    """
    Parse the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process raw eye tracking data.")
    parser.add_argument(
        "--dispersion-threshold",
        "-dit",
        type=float,
        default=DISPERSION_THRESHOLD_PX,
        help="The dispersion threshold for fixation detection in pixels.",
    )
    parser.add_argument(
        "--duration-threshold",
        "-dut",
        type=float,
        default=DURATION_THRESHOLD_NS,
        help="The duration threshold for fixation detection in nanoseconds.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function for processing the raw eye tracking data.
    """
    args = parse_arguments()
    dispersion_threshold = args.dispersion_threshold
    duration_threshold = args.duration_threshold

    # Remove old processed eye tracking data file
    print("➡️  Removing old eye tracking data files.")
    data_file_path = (
        f"{PROCESSED_EYE_TRACKING_DATA_PATH}/{PROCESSED_EYE_TRACKING_FILE_NAME}"
    )
    interpolated_data_file_path = f"{PROCESSED_EYE_TRACKING_DATA_PATH}/interpolated_{PROCESSED_EYE_TRACKING_FILE_NAME}"
    fixation_data_file_path = f"{PROCESSED_EYE_TRACKING_DATA_PATH}/fixation_{PROCESSED_EYE_TRACKING_FILE_NAME}"
    if os.path.exists(data_file_path):
        os.remove(data_file_path)
    if os.path.exists(interpolated_data_file_path):
        os.remove(interpolated_data_file_path)
    if os.path.exists(fixation_data_file_path):
        os.remove(fixation_data_file_path)

    raw_data = get_raw_data()
    processed_data = process_data(raw_data)
    interpolated_data = get_interpolated_data(processed_data)
    fixation_data = get_fixation_data(
        data=interpolated_data,
        dispersion_threshold_px=dispersion_threshold,
        duration_threshold_ns=duration_threshold,
    )

    # Save processed eye tracking data
    os.makedirs(PROCESSED_EYE_TRACKING_DATA_PATH, exist_ok=True)
    processed_data.to_csv(data_file_path, index=False)
    interpolated_data.to_csv(interpolated_data_file_path, index=False)
    fixation_data.to_csv(fixation_data_file_path, index=False)
    print(
        f"✅ Saved processed and interpolated eye tracking data to {data_file_path} and {interpolated_data_file_path}."
    )


if __name__ == "__main__":
    main()
