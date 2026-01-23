from typing import Optional, Union
from pathlib import Path
import json

def ensure_directory_exists(
    folder: Optional[Union[str, Path]] = None,
    file: Optional[Union[str, Path]] = None,
) -> bool:
    """If `folder` is provided, ensure it exists and return True.
    If `file` is provided, ensure its parent dir exists and return whether the file exists.
    Exactly one of `folder` or `file` must be provided.
    """
    if (folder is None) == (file is None):
        raise ValueError("Provide exactly one of 'folder' or 'file'.")

    if folder is not None:
        d = Path(folder)
        d.mkdir(parents=True, exist_ok=True)
        return True

    p = Path(file)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.exists()

def open_json(path: Union[str, Path]) -> dict:
    with open(path) as f:
        return json.load(f)
    
def save_json(data: dict, path: Union[str, Path], indent: int = 2):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)