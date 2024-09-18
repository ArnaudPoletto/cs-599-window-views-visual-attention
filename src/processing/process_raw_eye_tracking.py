import sys
from pathlib import Path

GLOBAL_DIR = Path(__file__).parent / ".." / ".."
sys.path.append(str(GLOBAL_DIR))

import os
import pandas as pd
from tqdm import tqdm
from typing import List

from src.utils.string import find_and_get_next_char_index
from src.config import (
    RAW_EYE_TRACKING_DATA_PATH,
    RAW_EYE_TRACKING_FRAME_WIDTH,
    RAW_EYE_TRACKING_FRAME_HEIGHT,
    PROCESSED_EYE_TRACKING_DATA_PATH,
)


def __delete_processed_files():
    """
    Delete the processed eye tracking data files.
    """
    print("⌛ Deleting already processed eye tracking data files...")

    for file_path in Path(PROCESSED_EYE_TRACKING_DATA_PATH).rglob(f"*.csv"):
        os.remove(file_path)

    print("✅ Already processed eye tracking data files deleted.")


def __get_src_file_paths() -> List[str]:
    """
    Get the source paths of the raw eye tracking data files.

    Returns:
        List[str]: List of the source paths of the raw eye tracking data files
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

    return file_paths


def __get_dst_file_paths(src_file_paths) -> List[str]:
    """
    Get the destination paths of the raw eye tracking data files.

    Args:
        src_file_paths (List[str]): List of the source paths of the raw eye tracking data files

    Returns:
        List[str]: List of the destination paths of the raw eye tracking data files
    """
    dst_file_paths = []
    for src_file_path in src_file_paths:

        # Extract global eye tracking data path
        global_eye_tracking_data_path = src_file_path.split("/raw")[0]

        # Extract experiment number
        experiment_number_index = find_and_get_next_char_index(
            src_file_path, "ETM Data EXP"
        )
        if experiment_number_index == -1:
            experiment_number_index = find_and_get_next_char_index(
                src_file_path, "ETM Data, EXP"
            )
        if experiment_number_index == -1:
            raise ValueError(
                f"❌ Could not find the experiment number in the path: {src_file_path}."
            )
        experiment_number = int(src_file_path[experiment_number_index])

        # Extract session number
        session_number_index = find_and_get_next_char_index(
            src_file_path, f"ETM Data EXP{experiment_number}, "
        )
        if session_number_index == -1:
            session_number_index = find_and_get_next_char_index(
                src_file_path, f"ETM Data, EXP{experiment_number}, "
            )
        if session_number_index == -1:
            raise ValueError(
                f"❌ Could not find the session number in the path: {src_file_path}."
            )
        session_number = int(src_file_path[session_number_index])

        # Extract participant id
        participant_id = int(src_file_path.split("/")[-1].split("_")[1][1:-1])

        # Extract extra file name data
        file_name_data_list = (
            src_file_path.split("/")[-1].replace(".csv", "").split("_")[2:]
        )
        file_name_data = "_".join(file_name_data_list)

        # Put in processed folder
        dst_file_path = global_eye_tracking_data_path
        dst_file_path += "/processed"
        dst_file_path += f"/experiment{experiment_number}"
        dst_file_path += f"/session{session_number}"
        dst_file_path += f"/participant{participant_id}_{file_name_data}.csv"

        dst_file_paths.append(dst_file_path)

    return dst_file_paths


def __process_data(
    raw_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Process the raw eye tracking data.

    Args:
        raw_data (pd.DataFrame): The raw eye tracking data

    Returns:
        pd.DataFrame: The processed eye tracking data
    """
    raw_data = raw_data.copy()

    # Remove false center gaze points
    raw_data = raw_data[
        (raw_data["GazeX"] != int(RAW_EYE_TRACKING_FRAME_WIDTH / 2))
        & (raw_data["GazeY"] != int(RAW_EYE_TRACKING_FRAME_HEIGHT / 2))
    ]

    # Delete entries with NaN values
    raw_data = raw_data.dropna()

    return raw_data


def __process_files(src_file_paths: List[str], dst_file_paths: List[str]):
    """
    Process the raw eye tracking data files.

    Args:
        src_file_paths (List[str]): List of the source paths of the raw eye tracking data files
        dst_file_paths (List[str]): List of the destination paths of the raw eye tracking data files
    """
    for src_file_path, dst_file_path in tqdm(
        zip(src_file_paths, dst_file_paths),
        total=len(src_file_paths),
        desc="⌛ Processing raw eye tracking data...",
        unit="file",
    ):
        # Read the raw eye tracking data
        raw_data = pd.read_csv(src_file_path, sep=";")

        # Process the raw eye tracking data
        processed_data = __process_data(raw_data)

        if processed_data.empty:
            print(f"⚠️ No data left after processing file {src_file_path}, skipping...")
            continue

        # Save the processed eye tracking data
        os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
        processed_data.to_csv(dst_file_path, index=False)

    print(f"✅ Eye tracking files processed.")


def __merge_files(participant_number: int, participant_file_paths: List[str]):
    if len(participant_file_paths) == 1:
        return

    # Read raw eye tracking data
    data_frames = []
    for participant_file_path in participant_file_paths:
        data_frames.append(pd.read_csv(participant_file_path))

    # Merge data frames
    merged_data = pd.concat(data_frames, ignore_index=True)

    # Extract global file path
    global_file_path = "/".join(participant_file_paths[0].split("/")[:-1])

    # Extract and merge extra file name data
    file_name_data_list = "_".join(
        [
            "_".join(
                participant_file_path.split("/")[-1].replace(".csv", "").split("_")[1:]
            )
            for participant_file_path in participant_file_paths
        ]
    )

    # Extract participant id
    merged_file_path = (
        f"{global_file_path}/participant{participant_number}_{file_name_data_list}.csv"
    )

    # Save the merged data
    merged_data.to_csv(merged_file_path, index=False)

    # Delete unmerged participant files
    for participant_file_path in participant_file_paths:
        os.remove(participant_file_path)


def __merge_participant_files():
    """
    Merge the participant raw eye tracking files.
    """
    print("⌛ Merging participant files...")
    for experiment_number in [1, 2]:
        for session_number in [1, 2]:
            folder_path = f"{PROCESSED_EYE_TRACKING_DATA_PATH}/experiment{experiment_number}/session{session_number}"
            file_paths = Path(folder_path).rglob(f"*.csv")
            file_paths = [file_path.resolve().as_posix() for file_path in file_paths]

            # Group files that belong to the same participant
            participant_files = {}
            for file_path in file_paths:
                participant_number = int(
                    file_path.split("/")[-1].split("_")[0].split("participant")[1]
                )

                if participant_number not in participant_files:
                    participant_files[participant_number] = []

                participant_files[participant_number].append(file_path)

            # Merge the files
            for participant_number, participant_file_paths in participant_files.items():
                __merge_files(participant_number, participant_file_paths)
    print("✅ Participant files merged.")


def main():
    """
    Main function for processing the raw eye tracking data.
    """
    __delete_processed_files()
    src_file_paths = __get_src_file_paths()
    dst_file_paths = __get_dst_file_paths(src_file_paths)
    __process_files(src_file_paths, dst_file_paths)
    __merge_participant_files()


if __name__ == "__main__":
    main()
