from typing import List, Tuple
from pathlib import Path


def get_files_recursive(
    folder_path: str,
    match_pattern: str,
) -> List[str]:
    """
    Get all file paths in the given folder path that match the given pattern recursively.

    Args:
        folder_path (str): Path to the folder
        match_pattern (str): Pattern to match the file names

    Returns:
        List[str]: List of file paths that match the given pattern
    """
    file_paths = list(Path(folder_path).rglob(match_pattern))
    file_paths = [file_path.resolve().as_posix() for file_path in file_paths]

    return file_paths


def get_set_str(experiment_id: int, set_id: int) -> str:
    """
    Get the set string based on the experiment and set id.

    Args:
        experiment_id (int): The experiment ID.
        set_id (int): The set ID.

    Raises:
        ValueError: If the experiment id is invalid.
        ValueError: If the set id is invalid.

    Returns:
        str: The set string
    """
    if experiment_id not in [1, 2]:
        raise ValueError(f"Invalid experiment id {experiment_id}.")
    if set_id not in [0, 1]:
        raise ValueError(f"Invalid set id {set_id}.")

    if experiment_id == 1:
        set_str = "images" if set_id == 1 else "videos"
    else:
        set_str = "clear" if set_id == 1 else "overcast"

    return set_str


def get_experiment_id_from_file_path(file_path: str) -> int:
    """
    Get the experiment id from the given file path.

    Args:
        file_path (str): The file path containing the experiment id.

    Returns:
        int: The experiment id
    """
    return int(file_path.split("/")[-3].split("experiment")[-1])


def get_set_id_from_file_path(file_path: str) -> str:
    """
    Get the set string from the given file path.

    Args:
        file_path (str): The file path containing the set string.

    Returns:
        str: The set string
    """
    set_str = file_path.split("/")[-2]
    set_str_to_id = {
        "videos": 0,
        "images": 1,
        "clear": 0,
        "overcast": 1,
    }

    return set_str_to_id[set_str]


def get_sequence_id_from_file_path(file_path: str) -> int:
    """
    Get the sequence id from the given file path.

    Args:
        file_path (str): The file path containing the sequence id.

    Returns:
        int: The sequence id
    """
    return int(file_path.split("/")[-1].split("scene")[-1].split(".")[0])


def get_ids_from_file_path(file_path: str) -> Tuple[int, int, int]:
    """
    Get the experiment, set, and sequence id from the given file path.

    Args:
        file_path (str): The file path containing the ids.

    Returns:
        int: The experiment ID.
        int: The set ID.
        int: The sequence ID.
    """
    experiment_id = get_experiment_id_from_file_path(file_path)
    set_id = get_set_id_from_file_path(file_path)
    sequence_id = get_sequence_id_from_file_path(file_path)

    return experiment_id, set_id, sequence_id
