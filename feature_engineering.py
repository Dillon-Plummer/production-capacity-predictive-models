# qualitylab/ml/feature_engineering.py

import pandas as pd

def add_recent_history(df: pd.DataFrame, window_days: int = 28) -> pd.DataFrame:
    """
    Takes a DataFrame with at least:
      - part_number
      - build_start_date (datetime)
      - build_time_days (float)
      - any number of columns named 'qty_of_defect_*'
    Returns a DataFrame with:
      - rolling 4-week sums for each defect column (named '<col>_4w_sum')
      - rolling 4-week average build time (build_time_4w_avg)
    Rows missing any of these new features will be dropped.
    """
    # 1) Sort and set the time index
    df = (
        df
        .sort_values(["part_number", "build_start_date"])
        .set_index("build_start_date")
    )

    # 2) Discover only the *original* defect columns (skip any _4w_sum)
    defect_cols = [
        c for c in df.columns
        if c.startswith("qty_of_defect_")
        and not c.endswith("_4w_sum")
    ]

    # 3) Compute 4-week rolling sums for each defect column
    for col in defect_cols:
        df[f"{col}_4w_sum"] = (
            df
            .groupby("part_number")[col]
            .rolling(f"{window_days}D")
            .sum()
            .reset_index(level=0, drop=True)
        )

    # 4) Compute 4-week rolling average build time
    df["build_time_4w_avg"] = (
        df
        .groupby("part_number")["build_time_days"]
        .rolling(f"{window_days}D")
        .mean()
        .reset_index(level=0, drop=True)
    )

    # 5) Restore build_start_date as a column
    df = df.reset_index()

    # 6) Drop any rows missing the newly created features
    required = [f"{col}_4w_sum" for col in defect_cols] + ["build_time_4w_avg"]
    df = df.dropna(subset=required)

    return df


def merge_downtime_features(df: pd.DataFrame, df_down: pd.DataFrame) -> pd.DataFrame:
    """
    For each build row in df, sum downtime_min and opportunity_cost
    from df_down where line matches and date falls within the build window.
    Also collect the failure modes seen in that window.
    """
    df = df.copy()
    df['downtime_min']    = 0.0
    df['opportunity_cost'] = 0.0
    df['failure_modes']   = [[] for _ in range(len(df))]
    df['failure_mode']    = 'NONE'
    
    # group downtime by line for efficiency
    down_by_line = {ln: sub for ln, sub in df_down.groupby('line')}
    
    for idx, row in df.iterrows():
        ln = row['line']
        if ln not in down_by_line:
            continue
        sub = down_by_line[ln]
        mask = (
            (sub['date'] >= row['build_start_date']) &
            (sub['date'] <= row['build_complete_date'])
        )
        hits = sub.loc[mask]
        
        # 1) sums
        df.at[idx, 'downtime_min']     = hits['downtime_min'].sum()
        df.at[idx, 'opportunity_cost'] = hits['opportunity_cost'].sum()
        
        # 2) collect all unique failure modes
        modes = hits['failure_mode'].dropna().unique().tolist()
        df.at[idx, 'failure_modes'] = modes
        
        # 3) pick a single mode (mode of the list, or NONE)
        if modes:
            # you could take the most frequent instead of first
            df.at[idx, 'failure_mode'] = modes[0]
        else:
            df.at[idx, 'failure_mode'] = 'NONE'
    
    return df
