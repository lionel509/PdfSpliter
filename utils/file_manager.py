# file_manager.py

import os
import shutil

def create_directory(path: str):
    """
    Creates a directory if it does not already exist.

    Args:
        path (str): Path to the directory.
    """
    os.makedirs(path, exist_ok=True)

def save_file(content: bytes, path: str):
    """
    Saves a file to the specified path.

    Args:
        content (bytes): File content to save.
        path (str): Path to save the file.
    """
    create_directory(os.path.dirname(path))
    with open(path, "wb") as file:
        file.write(content)

def load_file(path: str) -> bytes:
    """
    Loads the content of a file.

    Args:
        path (str): Path to the file.

    Returns:
        bytes: Content of the file.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as file:
        return file.read()

def delete_directory(path: str):
    """
    Deletes a directory and its contents.

    Args:
        path (str): Path to the directory.
    """
    if os.path.exists(path):
        shutil.rmtree(path)
