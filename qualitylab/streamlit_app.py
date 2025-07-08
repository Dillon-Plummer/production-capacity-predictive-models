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


# --- NEW: Explicit Column Validation Check ---
st.header("Data Validation")
required_features = set(qty_model.feature_names_in_) | set(bt_model.feature_names_in_) | set(defect_model.feature_names_in_)
data_columns = set(df.columns)

missing_features = required_features - data_columns

if missing_features:
    with st.expander("⚠️ Found Missing Columns in Loaded Data", expanded=True):
        st.error(
            "The following features are required by the models but were not found in the loaded data. "
            "They will be programmatically added as columns with a value of 0 to prevent the app from crashing."
        )
        st.json(sorted(list(missing_features)))
else:
    st.success("✅ Data columns successfully validated against model requirements.")

# --- FIX: Align DataFrame columns with all model features ---
# This loop prevents the KeyError by ensuring any feature a model needs
# exists in the DataFrame.
for feat in missing_features:
    df[feat] = 0
# --- End of Fix ---


X_qty = df[qty_model.feature_names_in_]
X_bt = df[bt_model.feature_names_in_]
X_def = df[defect_model.feature_names_in_]

pred_qty = qty_model.predict(X_qty)
pred_bt = bt_model.predict(X_bt)
pred_def = defect_model.predict(X_def)

pred_df = df[["part_number", "line", "qty_produced", "build_time_days", "build_start_date"]].copy()
pred_df["pred_qty"] = pred_qty
pred_df["pred_build_time"] = pred_bt

output_defect_cols = [c for c in defect_model.feature_names_in_ if c.startswith("qty_of_defect_")]

for i, col in enumerate(output_defect_cols):
    if col not in pred_df.columns:
        pred_df[col] = df[col]
    pred_df[f"pred_{col}"] = pred_def[:, i]

# --- New Feature: sidebar part number filter ---
part_options = sorted(pred_df["part_number"].unique())
selected_parts = st.sidebar.multiselect(
    "Filter by part number", part_options, default=part_options
)
filtered_df = pred_df[pred_df["part_number"].isin(selected_parts)]


tab_pred, tab_perf = st.tabs(["Predictions", "Model Performance"])

with tab_pred:
    st.subheader("Capacity Predictions")

    # --- New Feature: summary metrics ---
    tot_actual = int(filtered_df["qty_produced"].sum())
    tot_pred   = int(filtered_df["pred_qty"].sum())
    mean_time  = filtered_df["build_time_days"].mean()
    mean_pred  = filtered_df["pred_build_time"].mean()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Qty (Actual)", f"{tot_actual}")
    m2.metric("Total Qty (Pred)", f"{tot_pred}")
    m3.metric("Avg Build Time", f"{mean_time:.1f} d")
    m4.metric("Pred Build Time", f"{mean_pred:.1f} d")

    display_cols = [
        "part_number", "line", "qty_produced", "pred_qty",
        "build_time_days", "pred_build_time", "build_start_date"
    ]
    st.dataframe(filtered_df[display_cols])

    # --- New Feature: quantity trend chart ---
    chart_data = (
        filtered_df
        .sort_values("build_start_date")
        [["build_start_date", "qty_produced", "pred_qty"]]
        .set_index("build_start_date")
    )
    st.line_chart(chart_data, use_container_width=True)

with tab_perf:
    st.subheader("Defect Model Performance Over Time")

    for col in output_defect_cols:
        actual_col = col
        pred_col = f"pred_{col}"

        plot_data = filtered_df[['build_start_date', actual_col, pred_col]].copy()
        plot_data = plot_data.set_index('build_start_date')
        weekly_data = plot_data.resample('W').sum()

        melted_weekly = weekly_data.reset_index().melt(
            id_vars='build_start_date',
            value_vars=[actual_col, pred_col],
            var_name='Source',
            value_name='Count'
        )
        
        melted_weekly['Source'] = melted_weekly['Source'].map({
            actual_col: 'Actual',
            pred_col: 'Predicted'
        })

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=melted_weekly,
            x='build_start_date',
            y='Count',
            hue='Source',
            ax=ax,
            marker='o'
        )

        chart_title = col.replace("qty_of_defect_", "").replace("_", " ").title()
        ax.set_title(f"Performance for: {chart_title}", fontsize=16)
        ax.set_xlabel("Week")
        ax.set_ylabel("Total Weekly Count")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        st.markdown("---")
