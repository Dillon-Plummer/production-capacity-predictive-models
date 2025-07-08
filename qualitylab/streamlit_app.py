import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from paths import PROJECT_ROOT
from feature_engineering import add_recent_history, merge_downtime_features

st.set_page_config(layout="centered")
sns.set_theme(style="ticks")

@st.cache_data
def load_data() -> tuple[pd.DataFrame, list[str]]:
    """Load demo data and compute engineered features."""
    prod = pd.read_parquet(PROJECT_ROOT / "data" / "demo" / "production.parquet")
    down = pd.read_parquet(PROJECT_ROOT / "data" / "demo" / "downtime.parquet")
    df = add_recent_history(prod)
    df = merge_downtime_features(df, down)
    defect_cols = [
        c for c in df.columns if c.startswith("qty_of_defect_") and not c.endswith("_4w_sum")
    ]
    df["total_defects"] = df[defect_cols].sum(axis=1)
    qty = df["qty_produced"].replace(0, pd.NA)
    df["defect_rate"] = (df["total_defects"] / qty).fillna(0.0)
    return df, defect_cols

@st.cache_data
def load_models():
    """Load the latest trained models from disk."""
    model_dir = PROJECT_ROOT / "models"
    qty_model = joblib.load(sorted(model_dir.glob("build_quantity_model_*.pkl"))[-1])
    bt_model = joblib.load(sorted(model_dir.glob("build_time_model_*.pkl"))[-1])
    defect_model = joblib.load(sorted(model_dir.glob("defect_model_*.pkl"))[-1])
    return qty_model, bt_model, defect_model


df, defect_cols = load_data()
qty_model, bt_model, defect_model = load_models()

X_qty = df[["build_time_days", "build_time_4w_avg", "defect_rate", "downtime_min", "part_number", "line", "failure_mode"]]
X_bt = df[["build_time_4w_avg"] + [f"{c}_4w_sum" for c in defect_cols] + ["part_number", "line", "failure_mode"]]
X_def = df[["build_time_days", "build_time_4w_avg"] + [f"{c}_4w_sum" for c in defect_cols] + ["part_number", "line"]]

pred_qty = qty_model.predict(X_qty)
pred_bt = bt_model.predict(X_bt)
pred_def = defect_model.predict(X_def)

pred_df = df[["part_number", "line", "qty_produced", "build_time_days"]].copy()
pred_df["pred_qty"] = pred_qty
pred_df["pred_build_time"] = pred_bt
for i, col in enumerate(defect_cols):
    pred_df[f"pred_{col}"] = pred_def[:, i]


tab_pred, tab_perf = st.tabs(["Predictions", "Model Performance"])

with tab_pred:
    st.subheader("Capacity Predictions")
    st.dataframe(pred_df)

with tab_perf:
    st.subheader("Model Performance")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(x=df["qty_produced"], y=pred_qty, ax=ax1)
    ax1.set_xlabel("Actual quantity produced")
    ax1.set_ylabel("Predicted quantity")
    ax1.set_title("Build Quantity Model")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df["build_time_days"], y=pred_bt, ax=ax2)
    ax2.set_xlabel("Actual build time (days)")
    ax2.set_ylabel("Predicted build time (days)")
    ax2.set_title("Build Time Model")
    st.pyplot(fig2)

    first_def = defect_cols[0]
    fig3, ax3 = plt.subplots()
    sns.scatterplot(x=df[first_def], y=pred_def[:, 0], ax=ax3)
    ax3.set_xlabel(f"Actual {first_def}")
    ax3.set_ylabel(f"Predicted {first_def}")
    ax3.set_title("Defect Model (first defect type)")
    st.pyplot(fig3)
