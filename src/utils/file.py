from typing import List
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