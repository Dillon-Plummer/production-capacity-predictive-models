from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _ensure_dir(name: str) -> Path:
    d = PROJECT_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_data_dir() -> Path:
    """Return path to data directory, creating it if needed."""
    return _ensure_dir("data")


def get_model_dir() -> Path:
    """Return path to model directory, creating it if needed."""
    return _ensure_dir("models")


def get_output_dir() -> Path:
    """Return path to outputs directory, creating it if needed."""
    return _ensure_dir("outputs")
