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

X_qty = df[qty_model.feature_names_in_]
X_bt = df[bt_model.feature_names_in_]
X_def = df[defect_model.feature_names_in_]

pred_qty = qty_model.predict(X_qty)
pred_bt = bt_model.predict(X_bt)
pred_def = defect_model.predict(X_def)

pred_df = df[["part_number", "line", "qty_produced", "build_time_days", "build_start_date"]].copy()
pred_df["pred_qty"] = pred_qty
pred_df["pred_build_time"] = pred_bt
for i, col in enumerate(defect_cols):
    pred_df[f"pred_{col}"] = pred_def[:, i]


tab_pred, tab_perf = st.tabs(["Predictions", "Model Performance"])

with tab_pred:
    st.subheader("Capacity Predictions")
    st.dataframe(pred_df)

with tab_perf:
    st.subheader("Defect Model Performance Over Time")

    # Iterate through each defect type and create a line plot
    for col in defect_cols:
        actual_col = col
        pred_col = f"pred_{col}"

        # Prepare data for plotting: group by week and sum the counts
        plot_data = pred_df[['build_start_date', actual_col, pred_col]].copy()
        plot_data = plot_data.set_index('build_start_date')
        weekly_data = plot_data.resample('W').sum()

        # Melt the DataFrame to make it compatible with seaborn's 'hue'
        melted_weekly = weekly_data.reset_index().melt(
            id_vars='build_start_date',
            value_vars=[actual_col, pred_col],
            var_name='Source',
            value_name='Count'
        )
        
        # Rename for a cleaner legend
        melted_weekly['Source'] = melted_weekly['Source'].map({
            actual_col: 'Actual',
            pred_col: 'Predicted'
        })

        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(
            data=melted_weekly,
            x='build_start_date',
            y='Count',
            hue='Source',
            ax=ax,
            marker='o'
        )

        # Prettify the plot
        chart_title = col.replace("qty_of_defect_", "").replace("_", " ").title()
        ax.set_title(f"Performance for: {chart_title}", fontsize=16)
        ax.set_xlabel("Week")
        ax.set_ylabel("Total Weekly Count")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        st.pyplot(fig)
        st.markdown("---")
