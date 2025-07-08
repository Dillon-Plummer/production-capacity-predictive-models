import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path
from paths import PROJECT_ROOT

# Configure page
st.set_page_config(layout="centered")
sns.set_theme(style="ticks")

@st.cache_data
def load_demo_data() -> pd.DataFrame:
    """Load and prepare the demo production data."""
    data_path = PROJECT_ROOT / "data" / "demo" / "production_demo_data.xlsx"
    df = pd.read_excel(data_path)
    df["build_time_days"] = (
        df["build_complete_date"] - df["build_start_date"]
    ).dt.total_seconds() / 86400
    defect_cols = [c for c in df.columns if c.startswith("qty_of_defect_")]
    df["total_defects"] = df[defect_cols].sum(axis=1)
    df["defect_rate"] = df["total_defects"] / df["qty_produced"]
    return df[["qty_produced", "build_time_days", "defect_rate"]]

df = load_demo_data()

# Sidebar controls
st.sidebar.header("Options")
overlap = st.sidebar.slider("Data overlap", 0.0, 1.0, 1.0, step=0.1)
sample_size = st.sidebar.number_input(
    "Sample size", min_value=10, max_value=len(df), value=min(200, len(df)), step=10
)

st.header("New Supplier Inputs")
new_qty = st.number_input("Projected quantity", value=1000)
new_bt = st.number_input("Projected build time (days)", value=1.0)
new_defect = st.number_input(
    "Projected defect rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01
)

# Prepare data for plotting
sample_n = int(sample_size * overlap)
base = df.sample(sample_n, random_state=0).copy()
base["supplier"] = "Existing"
new_row = {
    "qty_produced": new_qty,
    "build_time_days": new_bt,
    "defect_rate": new_defect,
    "supplier": "New",
}
plot_df = pd.concat([base, pd.DataFrame([new_row])], ignore_index=True)

# Pair plot
fig = sns.pairplot(plot_df, hue="supplier", diag_kind="hist")
st.pyplot(fig)
