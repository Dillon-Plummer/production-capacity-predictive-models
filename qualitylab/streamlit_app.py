import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from lime.lime_tabular import LimeTabularExplainer
from feature_engineering import add_recent_history, merge_downtime_features
from upsetplot import from_indicators, UpSet
from paths import PROJECT_ROOT, get_model_dir, get_output_dir
import io
import zipfile



# Page config & figure sizing
st.set_page_config(layout="centered")
sns.set(rc={"figure.figsize": (16, 9)})

# Session-state defaults
if "exports" not in st.session_state:
    st.session_state.exports = {}
if "rolling_window" not in st.session_state:
    st.session_state.rolling_window = 28
# The initializations for user-uploaded files are no longer needed.
if "demo_prod_files" not in st.session_state:
    st.session_state.demo_prod_files = []
if "demo_down_files" not in st.session_state:
    st.session_state.demo_down_files = []
if "demo_plan_file" not in st.session_state:
    st.session_state.demo_plan_file = None

# Ensure project root on path
project_root = PROJECT_ROOT
sys.path.insert(0, str(project_root))

def get_latest_model(pattern: str) -> Path:
    files = list(get_model_dir().glob(pattern))
    if not files:
        st.error(f"No models found for pattern `{pattern}`")
        st.stop()
    return max(files, key=lambda p: p.stat().st_mtime)

st.title("QualityLab ML Dashboard (Seaborn)")

# Helper to trigger a rerun with demo data loaded
def _flag_use_demo():
    st.session_state.use_demo = True
    st.experimental_rerun()

# Load demo data before widgets are instantiated if flagged
if st.session_state.get("use_demo"):
    demo_dir = PROJECT_ROOT / "data" / "demo"
    st.session_state.demo_prod_files = [demo_dir / "production_demo_data.xlsx"]
    st.session_state.demo_down_files = [demo_dir / "downtime_demo_data.xlsx"]
    st.session_state.demo_plan_file = demo_dir / "build_plan_demo.xlsx"
    st.session_state.uploaded = True
    st.session_state.use_demo = False

# — Sidebar: Load Demo Data —
if "uploaded" not in st.session_state:
    st.session_state.uploaded = False

# The original file upload form has been commented out as requested.
# with st.sidebar.form("upload_form"):
#     st.header("Upload Data")
#     up_prod = st.file_uploader(
#         "Production sheets", type=["xlsx", "xls", "csv"],
#         accept_multiple_files=True, key="prod_files"
#     )
#     up_down = st.file_uploader(
#         "Downtime sheets", type=["xlsx", "xls", "csv"],
#         accept_multiple_files=True, key="down_files"
#     )
#     up_plan = st.file_uploader(
#         "Build Plan", type=["xlsx", "xls", "csv"],
#         accept_multiple_files=False, key="plan_file"
#     )
#     submitted = st.form_submit_button("Submit")
#     st.form_submit_button("Use Demo Data", on_click=_flag_use_demo)
#     if submitted:
#         if not (up_prod and up_down and up_plan):
#             st.warning("Please upload production, downtime AND plan files.")
#         else:
#             st.session_state.uploaded = True

# Replacement UI: A single button to load the demo data.
st.sidebar.header("Load Data")
st.sidebar.button("Load Demo Data", on_click=_flag_use_demo, use_container_width=True)


if not st.session_state.uploaded:
    st.sidebar.info("Click 'Load Demo Data' to begin analysis.")
    st.stop()

# Rolling window slider
st.sidebar.write("---")
st.session_state.rolling_window = st.sidebar.slider(
    "Rolling window (days)",
    min_value=7,
    max_value=90,
    value=st.session_state.rolling_window,
    step=1,
)
rolling_window = st.session_state.rolling_window

# Master export button
if st.session_state.exports:
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for fname, data in st.session_state.exports.items():
            zf.writestr(fname, data)
    zip_buf.seek(0)
    output_dir = get_output_dir()
    with open(output_dir / "all_exports.zip", "wb") as f:
        f.write(zip_buf.getvalue())
    st.sidebar.download_button(
        "Download All Exports", zip_buf.getvalue(), "all_exports.zip", mime="application/zip"
    )

# — Ingest & clean production data —
# MODIFIED: Removed reference to user-uploaded prod_files. Now only uses demo files.
# prod_sources = st.session_state.get("prod_files", []) or st.session_state.demo_prod_files
prod_sources = st.session_state.demo_prod_files
raw_prod = []
for up in prod_sources:
    if isinstance(up, Path) or up.name.lower().endswith(("xlsx", "xls")):
        raw_prod.append(pd.read_excel(up, engine="openpyxl"))
    else:
        raw_prod.append(pd.read_csv(up))
df_prod = pd.concat(raw_prod, ignore_index=True)
df_prod.columns = (
    df_prod.columns
           .str.strip()
           .str.lower()
           .str.replace(r"[ _]+", "_", regex=True)
)
for c in ("build_start_date", "build_complete_date"):
    df_prod[c] = pd.to_datetime(df_prod[c], errors="coerce")
df_prod = df_prod.dropna(subset=["build_start_date", "build_complete_date"])
df_prod["build_time_days"] = (
    df_prod["build_complete_date"] - df_prod["build_start_date"]
).dt.total_seconds() / 86400

if "line" not in df_prod.columns:
    st.error("Missing 'line' in production data")
    st.stop()
df_prod["line"] = df_prod["line"].astype(str).str.strip().str.upper()
valid_lines = df_prod["line"].unique().tolist()

if "part_number" not in df_prod.columns:
    st.error("Missing 'part_number' in production data")
    st.stop()
df_prod["part_number"] = df_prod["part_number"].astype(str).str.strip()

orig_defs = [c for c in df_prod.columns if c.startswith("qty_of_defect_")]
if not orig_defs:
    st.error("No defect columns found in production data")
    st.stop()
for c in orig_defs:
    df_prod[c] = pd.to_numeric(df_prod[c], errors="coerce").fillna(0)

# — Feature engineering on prod history —
df_fe = add_recent_history(df_prod, window_days=rolling_window)

# — Ingest & clean downtime data —
# MODIFIED: Removed reference to user-uploaded down_files. Now only uses demo files.
# down_sources = st.session_state.get("down_files", []) or st.session_state.demo_down_files
down_sources = st.session_state.demo_down_files
raw_down = []
for up in down_sources:
    if isinstance(up, Path) or up.name.lower().endswith(("xlsx", "xls")):
        raw_down.append(pd.read_excel(up, engine="openpyxl"))
    else:
        raw_down.append(pd.read_csv(up))
df_down = pd.concat(raw_down, ignore_index=True)
df_down.columns = (
    df_down.columns
           .str.strip()
           .str.lower()
           .str.replace(r"[ _]+", "_", regex=True)
)
if "date" not in df_down.columns:
    st.error("Missing 'date' in downtime data")
    st.stop()
df_down["date"] = pd.to_datetime(df_down["date"], errors="coerce")
df_down = df_down.dropna(subset=["date"])
if "line" not in df_down.columns:
    st.error("Missing 'line' in downtime data")
    st.stop()
df_down["line"] = df_down["line"].astype(str).str.strip().str.upper()
mask_bad = ~df_down["line"].isin(valid_lines)
if mask_bad.any():
    st.warning(f"Dropping {mask_bad.sum()} downtime rows on lines not in production")
    df_down = df_down.loc[~mask_bad].copy()

if "failure_mode" in df_down.columns:
    df_down["failure_mode"] = (
        df_down["failure_mode"]
          .astype(str)
          .str.strip()
          .str.upper()
          .str.replace(r"^\d+\s*-\s*", "", regex=True)
    )
else:
    df_down["failure_mode"] = "NONE"

# — Merge downtime into history features —
df_fe = merge_downtime_features(df_fe, df_down)

# — Save a copy of full, unfiltered history —
df_fe_full = df_fe.copy()

# — Dynamically discover defect columns & defect-4w sums —
orig_defs = [
    c for c in df_fe.columns
    if c.startswith("qty_of_defect_") and not c.endswith("_4w_sum")
]
defect_4w = [f"{c}_4w_sum" for c in orig_defs]

df_fe["total_defects"] = df_fe[orig_defs].sum(axis=1)
df_fe["defect_rate"]    = df_fe["total_defects"] / df_fe["qty_produced"]

# — Ingest & clean build plan —
# MODIFIED: Removed reference to user-uploaded plan_file. Now only uses demo file.
# up = st.session_state.get("plan_file", None) or st.session_state.demo_plan_file
up = st.session_state.demo_plan_file
if isinstance(up, Path) or up.name.lower().endswith(("xlsx", "xls")):
    df_plan = pd.read_excel(up, engine="openpyxl")
else:
    df_plan = pd.read_csv(up)
df_plan.columns = (
    df_plan.columns
           .str.strip()
           .str.lower()
           .str.replace(r"[ _]+", "_", regex=True)
)
df_plan["plan_start_date"] = pd.to_datetime(df_plan["plan_start_date"], errors="coerce")
if "plan_end_date" in df_plan.columns:
    df_plan["plan_end_date"] = pd.to_datetime(df_plan["plan_end_date"], errors="coerce")
else:
    df_plan["plan_end_date"] = df_plan["plan_start_date"]
auto_bt = (df_plan["plan_end_date"] - df_plan["plan_start_date"]).dt.total_seconds() / 86400
if "build_time_days" in df_plan.columns:
    df_plan["build_time_days"] = df_plan["build_time_days"].fillna(auto_bt)
else:
    df_plan["build_time_days"] = auto_bt

df_plan = df_plan.dropna(subset=[
    "part_number",
    "planned_qty",
    "plan_start_date",
    "plan_end_date",
    "build_time_days",
])
if "line" not in df_plan.columns:
    st.error("Missing 'line' in build plan")
    st.stop()
df_plan["line"] = df_plan["line"].astype(str).str.strip().str.upper()
df_plan["part_number"] = df_plan["part_number"].astype(str).str.strip()


st.write("🗓️ Production date range:",
         df_prod["build_start_date"].min().date(), "→", df_prod["build_start_date"].max().date())
st.write("🗓️ Plan start dates:",
         df_plan["plan_start_date"].min().date(), "→", df_plan["plan_start_date"].max().date())

# — Build plan features dynamically, include real 4-week defect sums & failure_mode —
def make_plan_features(r):
    pn    = r["part_number"]
    st_dt = r["plan_start_date"]
    ln    = r["line"]

    hist = df_prod[
        (df_prod["part_number"] == pn)
        & (df_prod["build_start_date"] >= st_dt - pd.Timedelta(days=rolling_window))
        & (df_prod["build_start_date"] < st_dt)
    ]

    defect_4w_vals = {f"{c}_4w_sum": hist[c].sum() for c in orig_defs}
    bt_4w = hist["build_time_days"].mean() if not hist.empty else 0.0

    segment = df_down[
        (df_down.line == ln) &
        (df_down.date.between(st_dt, r.plan_end_date))
    ]

    dt_sum = segment.downtime_min.sum()
    modes = segment.failure_mode.dropna().unique().tolist()
    mode_list = ", ".join(modes) if modes else "NONE"

    sub = df_down[df_down["line"] == ln]
    fm_mask = (sub["date"] >= st_dt) & (sub["date"] <= r["plan_end_date"])
    matched = sub.loc[fm_mask, "failure_mode"]

    if not matched.empty:
        most_common = matched.mode()
        if not most_common.empty:
            failure_for_row = most_common.iloc[0]
        else:
            failure_for_row = matched.iloc[0]
    else:
        failure_for_row = "NONE"
    failure_for_row = str(failure_for_row).strip().upper()

    total_defects = hist[orig_defs].sum().sum()
    total_qty     = hist.get("qty_produced", pd.Series(dtype=float)).sum()
    defect_rate   = total_defects / total_qty if total_qty > 0 else 0.0

    feat = {
        "part_number":       pn,
        "line":              ln,
        "planned_qty":       r["planned_qty"],
        "plan_start_date":   st_dt,
        "plan_end_date":     r["plan_end_date"],
        "build_time_days":   r["build_time_days"],
        "build_time_4w_avg": bt_4w,
        "defect_rate":       defect_rate,
        "downtime_min":      dt_sum,
        "failure_mode":      failure_for_row,
        "failure_modes":     mode_list,
        **defect_4w_vals
    }
    feat.update(defect_4w_vals)
    return feat

if not df_plan.empty:
    df_plan_feats = pd.DataFrame([make_plan_features(r) for _, r in df_plan.iterrows()])
else:
    df_plan_feats = pd.DataFrame(columns=[
        "part_number","line","planned_qty","plan_start_date","plan_end_date",
        "build_time_days","build_time_4w_avg","defect_rate","downtime_min",
        "failure_mode","failure_modes", *defect_4w
    ])
if "failure_mode" not in df_plan_feats.columns:
    df_plan_feats["failure_mode"] = "NONE"
else:
    df_plan_feats["failure_mode"] = df_plan_feats["failure_mode"].astype(str).str.strip().str.upper()

# ─────────────────────────────────────────────────────────────────────────────

# — Load models & extract expected feature names —
build_model   = joblib.load(get_latest_model("build_time_model_*.pkl"))
defect_model  = joblib.load(get_latest_model("defect_model_*.pkl"))
qty_model     = joblib.load(get_latest_model("build_quantity_model_*.pkl"))

bt_feats          = list(build_model.feature_names_in_)
tq_feats          = list(qty_model.feature_names_in_)
def_model_feats   = list(defect_model.feature_names_in_)

for c in bt_feats:
    if c not in df_fe.columns:
        df_fe[c] = 0.0
df_fe["pred_build_time"] = build_model.predict(df_fe[bt_feats])

for c in tq_feats:
    if c not in df_fe.columns:
        df_fe[c] = 0.0
df_fe["pred_quantity"] = qty_model.predict(df_fe[tq_feats])

for c in def_model_feats:
    if c not in df_fe.columns:
        df_fe[c] = 0.0
preds = defect_model.predict(df_fe[def_model_feats])
for i, col in enumerate(orig_defs):
    df_fe[f"pred_{col}"] = preds[:, i]

if not df_plan_feats.empty:
    num_bt_feats = [c for c in bt_feats if c not in ("part_number", "line", "failure_mode")]
    cat_bt_feats = ["part_number", "line", "failure_mode"]

    for c in num_bt_feats:
        if c not in df_plan_feats.columns:
            df_plan_feats[c] = 0.0
        else:
            df_plan_feats[c] = pd.to_numeric(df_plan_feats[c], errors="coerce").fillna(0.0)

    for c in cat_bt_feats:
        if c not in df_plan_feats.columns:
            df_plan_feats[c] = "NONE"
        else:
            df_plan_feats[c] = df_plan_feats[c].astype(str).str.strip().str.upper()

    df_plan_feats["pred_build_time"] = build_model.predict(df_plan_feats[bt_feats])

    num_tq_feats = [c for c in tq_feats if c not in ("part_number", "line", "failure_mode")]
    cat_tq_feats = ["part_number", "line", "failure_mode"]

    for c in num_tq_feats:
        if c not in df_plan_feats.columns:
            df_plan_feats[c] = 0.0
        else:
            df_plan_feats[c] = pd.to_numeric(df_plan_feats[c], errors="coerce").fillna(0.0)

    for c in cat_tq_feats:
        if c not in df_plan_feats.columns:
            df_plan_feats[c] = "NONE"
        else:
            df_plan_feats[c] = df_plan_feats[c].astype(str).str.strip().str.upper()

    df_plan_feats["pred_qty"] = qty_model.predict(df_plan_feats[tq_feats])
else:
    df_plan_feats["pred_build_time"] = pd.Series(dtype=float)
    df_plan_feats["pred_qty"]        = pd.Series(dtype=float)

# The rest of the script continues as before...
# (The LIME, filtering, and plotting code does not need to be changed as it
# operates on the DataFrames created in the steps above)
# ...
