# =========================================================
# ⚡ Energy Forecasting & Monitoring Dashboard
# Production-ready Streamlit App
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import mlflow
from pathlib import Path
from scipy.stats import ks_2samp

# =========================================================
# Page Config
# =========================================================
st.set_page_config(
    page_title="Energy Forecast Dashboard",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Energy Forecasting & Monitoring Platform")

# =========================================================
# Paths
# =========================================================
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "energy_data.csv"
MODEL_PATH = ROOT / "xgb_model.pkl"

# =========================================================
# Loaders
# =========================================================
@st.cache_data
def load_baseline():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df["rolling7"] = df["Energy"].rolling(7).mean()
    return df

@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_runs():
    try:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        runs = mlflow.search_runs()

        if runs.empty:
            return pd.DataFrame()

        cols = ["run_id", "metrics.MAE", "metrics.RMSE", "metrics.MAPE"]
        runs = runs[[c for c in cols if c in runs.columns]]

        runs.rename(
            columns={
                "metrics.MAE": "MAE",
                "metrics.RMSE": "RMSE",
                "metrics.MAPE": "MAPE",
            },
            inplace=True,
        )

        return runs.sort_values("MAE")

    except Exception:
        return pd.DataFrame()

# =========================================================
# Utils
# =========================================================
def feature_drift(base, current):
    results = []
    numeric_cols = base.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        if col not in current.columns:
            continue

        stat, p = ks_2samp(base[col].dropna(), current[col].dropna())

        if p < 0.01:
            level = "🔴 HIGH"
        elif p < 0.05:
            level = "🟡 MEDIUM"
        else:
            level = "🟢 LOW"

        results.append([col, stat, p, level])

    return pd.DataFrame(results, columns=["feature", "ks_stat", "p_value", "drift"])

def local_predict(file):
    model = load_model()
    if model is None:
        st.error("Model file not found.")
        return None

    df = pd.read_csv(file)
    df["prediction"] = model.predict(df)
    return df

# =========================================================
# Load Data & Model
# =========================================================
baseline_df = load_baseline()
runs_df = load_runs()
model = load_model()

# =========================================================
# Sidebar Controls
# =========================================================
st.sidebar.header("⚙ Controls")
date_range = st.sidebar.date_input(
    "Date Range",
    [baseline_df["Date"].min(), baseline_df["Date"].max()],
)
show_rolling = st.sidebar.checkbox("Show 7-day average", value=True)
filtered_df = baseline_df[
    (baseline_df["Date"] >= pd.to_datetime(date_range[0])) &
    (baseline_df["Date"] <= pd.to_datetime(date_range[1]))
]

# =========================================================
# KPI Cards
# =========================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Records", len(filtered_df))
c2.metric("Avg Energy", round(filtered_df["Energy"].mean(), 2))
c3.metric("Max Energy", round(filtered_df["Energy"].max(), 2))
c4.metric("Min Energy", round(filtered_df["Energy"].min(), 2))

st.divider()

# =========================================================
# Tabs
# =========================================================
tabs = st.tabs([
    "📈 Historical",
    "🔮 Forecast",
    "🚀 Predictions",
    "📉 Drift",
    "📊 Experiments",
])

# =========================================================
# TAB 1 — Historical
# =========================================================
with tabs[0]:
    st.subheader("Energy Demand Trend")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(filtered_df["Date"], filtered_df["Energy"], label="Energy")
    if show_rolling:
        ax.plot(filtered_df["Date"], filtered_df["rolling7"], label="7 Day Avg")
    ax.set_xlabel("Date")
    ax.set_ylabel("Energy")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(filtered_df.tail(), use_container_width=True)

# =========================================================
# TAB 2 — Forecast
# =========================================================
with tabs[1]:
    st.subheader("7 Day Energy Forecast")
    if model is None:
        st.warning("Model not found")
    else:
        future_dates = pd.date_range(start=baseline_df["Date"].max(), periods=7)
        future_df = pd.DataFrame({
            "Temperature": np.random.normal(25, 2, 7),
            "lag1": np.random.normal(320, 10, 7),
            "lag7": np.random.normal(300, 10, 7),
            "rolling7": np.random.normal(310, 5, 7),
            "dayofweek": future_dates.dayofweek
        })
        forecast = model.predict(future_df)
        forecast_df = pd.DataFrame({"Date": future_dates, "Forecast": forecast})
        st.line_chart(forecast_df.set_index("Date"))

# =========================================================
# TAB 3 — Predictions
# =========================================================
with tabs[2]:
    st.subheader("Interactive Prediction")
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Temperature", 0, 40, 25)
        lag1 = st.slider("Yesterday Demand", 200, 500, 320)
        lag7 = st.slider("Last Week Demand", 200, 500, 300)
    with col2:
        rolling7 = st.slider("7-Day Avg", 200, 500, 310)
        day = st.selectbox("Day of Week", list(range(7)))

    if st.button("Predict Demand"):
        input_df = pd.DataFrame({
            "Temperature":[temp],
            "lag1":[lag1],
            "lag7":[lag7],
            "rolling7":[rolling7],
            "dayofweek":[day]
        })
        pred = model.predict(input_df)[0]
        st.success(f"Predicted Energy Demand: {round(pred,2)}")

    st.divider()
    st.subheader("Batch CSV Prediction")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        preds = local_predict(uploaded)
        if preds is not None:
            st.dataframe(preds.head(), use_container_width=True)
            fig, ax = plt.subplots()
            ax.plot(preds["prediction"])
            ax.set_title("Prediction Output")
            st.pyplot(fig)

# =========================================================
# TAB 4 — Drift Monitoring
# =========================================================
with tabs[3]:
    st.subheader("Feature Drift Detection")
    drift_file = st.file_uploader("Upload Production Data", type=["csv"], key="drift")
    if drift_file:
        current_df = pd.read_csv(drift_file)
        drift_df = feature_drift(baseline_df, current_df)
        st.dataframe(drift_df, use_container_width=True)
        fig, ax = plt.subplots()
        ax.bar(drift_df["feature"], drift_df["ks_stat"])
        ax.set_xticks(range(len(drift_df["feature"])))
        ax.set_xticklabels(drift_df["feature"], rotation=45)
        ax.set_title("Feature Drift Score")
        st.pyplot(fig)

# =========================================================
# TAB 5 — MLflow Experiments
# =========================================================
with tabs[4]:
    st.subheader("MLflow Experiment Tracking")
    if runs_df.empty:
        st.info("No MLflow runs found.")
    else:
        st.dataframe(runs_df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.bar_chart(runs_df.set_index("run_id")["MAE"])
        with c2:
            st.bar_chart(runs_df.set_index("run_id")["RMSE"])

# =========================================================
# Feature Importance (Upgraded)
# =========================================================
st.divider()
st.subheader("🔎 Model Feature Importance")

if model is not None and hasattr(model, "feature_importances_"):
    features = ["Temperature","lag1","lag7","rolling7","dayofweek"]
    importance = model.feature_importances_
    fi_df = pd.DataFrame({"feature": features, "importance": importance}).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8,4))
    colors = ["#1f77b4" if i < 0.6*fi_df["importance"].max() else "#ff7f0e" for i in fi_df["importance"]]
    ax.barh(fi_df["feature"], fi_df["importance"], color=colors)

    for i, (imp, feat) in enumerate(zip(fi_df["importance"], fi_df["feature"])):
        ax.text(imp + 0.01*fi_df["importance"].max(), i, f"{imp*100:.1f}%", va='center')

    ax.set_xlabel("Importance Score")
    ax.set_title("Model Feature Importance")
    st.pyplot(fig)
else:
    st.info("Model or feature importance not available.")

# =========================================================
# Footer
# =========================================================
st.divider()
st.caption("⚡ Built with Streamlit • Energy Forecasting MLOps Dashboard")