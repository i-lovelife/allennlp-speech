from pathlib import Path
def to_hdf5_path(file_path: str) -> str:
    if isinstance(file_path, Path):
        file_path = str(file_path)
    