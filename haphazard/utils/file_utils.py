"""
haphazard.utils.file_utils
--------------------------
File and path utility functions.

This module provides helper functions for discovering, validating, and
managing files and directories within the project. Useful for
dataset/model loading, configuration discovery, and other filesystem tasks.

Notes
-----
- Functions generally return absolute paths or validated objects.
"""

from pathlib import Path


# -------------------------------------------------------------------------
# File discovery function
# -------------------------------------------------------------------------
def find_file(base_path: str | Path, file_path: str) -> str:
    """
    Recursively search for a file (or relative path) under a base directory.

    Parameters
    ----------
    base_path : str | Path
        Directory under which the file search is performed.
    file_path : str
        File name or relative path to search for.

    Returns
    -------
    : str
        Absolute path of the first matching file found.

    Raises
    ------
    FileNotFoundError
        If the file is not found under the `base_path`.

    Example
    -------
    >>> find_file("/home/user/projects", "config.yaml")
    '/home/user/projects/config.yaml'
    """
    base_path = Path(base_path).resolve()
    for f in base_path.rglob(file_path):
        return str(f)  # Returns the first found instance
    raise FileNotFoundError(f"{file_path} not found under {base_path}")
