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


def get_session_str(experiment_id: int, session_id: int) -> str:
    """
    Get the session string based on the experiment and session id.

    Args:
        experiment_id (int): The experiment ID.
        session_id (int): The session ID.

    Raises:
        ValueError: If the experiment id is invalid
        ValueError: If the session id is invalid

    Returns:
        str: The session string
    """
    if experiment_id not in [1, 2]:
        raise ValueError(f"Invalid experiment id {experiment_id}.")
    if session_id not in [1, 2]:
        raise ValueError(f"Invalid session id {session_id}.")

    if experiment_id == 1:
        session_str = "images" if session_id == 1 else "videos"
    else:
        session_str = "clear" if session_id == 1 else "overcast"

    return session_str


def get_experiment_id_from_file_path(file_path: str) -> int:
    """
    Get the experiment id from the given file path.

    Args:
        file_path (str): The file path containing the experiment id.

    Returns:
        int: The experiment id
    """
    return int(file_path.split("/")[-3].split("experiment")[-1])


def get_session_id_from_file_path(file_path: str) -> str:
    """
    Get the session string from the given file path.

    Args:
        file_path (str): The file path containing the session string.

    Returns:
        str: The session string
    """
    session_str = file_path.split("/")[-2]
    session_str_to_id = {
        "images": 1,
        "videos": 2,
        "clear": 1,
        "overcast": 2,
    }

    return session_str_to_id[session_str]


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
    Get the experiment, session, and sequence id from the given file path.

    Args:
        file_path (str): The file path containing the ids.

    Returns:
        int: The experiment ID.
        int: The session ID.
        int: The sequence ID.
    """
    experiment_id = get_experiment_id_from_file_path(file_path)
    session_id = get_session_id_from_file_path(file_path)
    sequence_id = get_sequence_id_from_file_path(file_path)

    return experiment_id, session_id, sequence_id
