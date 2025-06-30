import pandas as pd
from pathlib import Path

def _read_file(path: Path) -> pd.DataFrame:
    if str(path).lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(path, engine="openpyxl")
    else:
        return pd.read_csv(path)

def read_production_data(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate production sheets from given paths."""
    frames = [_read_file(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[ _]+", "_", regex=True)
    )
    for col in ("build_start_date", "build_complete_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    if "build_start_date" in df.columns and "build_complete_date" in df.columns:
        df = df.dropna(subset=["build_start_date", "build_complete_date"])
        df["build_time_days"] = (
            df["build_complete_date"] - df["build_start_date"]
        ).dt.total_seconds() / 86400
    if "line" in df.columns:
        df["line"] = df["line"].astype(str).str.strip().str.upper()
    if "part_number" in df.columns:
        df["part_number"] = df["part_number"].astype(str).str.strip()
    defect_cols = [c for c in df.columns if c.startswith("qty_of_defect_")]
    for c in defect_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df

def read_downtime_data(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate downtime logs from given paths."""
    frames = [_read_file(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r"[ _]+", "_", regex=True)
    )
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
    if "line" in df.columns:
        df["line"] = df["line"].astype(str).str.strip().str.upper()
    if "failure_mode" in df.columns:
        df["failure_mode"] = (
            df["failure_mode"]
              .astype(str)
              .str.strip()
              .str.upper()
              .str.replace(r"^\d+\s*-\s*", "", regex=True)
        )
    else:
        df["failure_mode"] = "NONE"
    for col in ("downtime_min", "opportunity_cost"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df
