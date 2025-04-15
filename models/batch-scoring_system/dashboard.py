import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import folium
from streamlit_folium import st_folium
import xgboost as xgb
import numpy as np
import gc

# Project directory structure
BASE_DIR = Path(__file__).resolve().parent.parent.parent
FEATURE_STORE_DIR = BASE_DIR / "models" / "batch-scoring_system" / "feature_store"
PREDICTIONS_DIR = BASE_DIR / "models" / "batch-scoring_system" / "predictions"
MODELS_DIR = BASE_DIR / "models" / "batch-scoring_system" / "models_and_metadata"


# Load data with error handling and optimized column selection
@st.cache_data
def load_feature_store():
    feature_file = FEATURE_STORE_DIR / "features.parquet"
    try:
        # Load columns, including the precomputed lagged and rolling features
        columns_to_load = [
            "hourly_timestamp",
            "PULocationID",
            "ride_count",
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "is_weekend",
            "is_holiday",
            "is_rush_hour",
            "temperature",
            "humidity",
            "wind_speed",
            "cloud_cover_numeric",
            "amount_of_precipitation",
            "is_downtown",
            "lag_1_ride_count",
            "lag_2_ride_count",
            "lag_24_ride_count",
            "rolling_mean_3h",
            "rolling_mean_24h",
        ]
        df = pd.read_parquet(feature_file, columns=columns_to_load)
        df["hourly_timestamp"] = pd.to_datetime(df["hourly_timestamp"])
        return df
    except Exception as e:
        st.error(f"Error loading feature store: {e}")
        return None


@st.cache_data
def load_predictions():
    prediction_file = PREDICTIONS_DIR / "predictions_20190401_000000.parquet"
    try:
        df = pd.read_parquet(prediction_file)
        df["hourly_timestamp"] = pd.to_datetime(df["hourly_timestamp"])
        return df
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return None


@st.cache_data
def get_latest_model_and_metadata(model_name="xgboost"):
    """Load the latest model and metadata files based on timestamps with memory optimization."""
    try:
        # Find the latest model and metadata files
        model_files = list(MODELS_DIR.glob(f"{model_name}_*.joblib"))
        metadata_files = list(MODELS_DIR.glob(f"{model_name}_metadata_*.json"))

        if not model_files or not metadata_files:
            raise FileNotFoundError(
                "No model or metadata files found in the specified directory."
            )

        latest_model_file = max(model_files, key=lambda x: x.stat().st_mtime)
        latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)

        # Load metadata first (smaller file, less memory-intensive)
        with open(latest_metadata_file, "r") as f:
            metadata = json.load(f)

        # Garbage collection to free memory before loading the model
        gc.collect()

        # Load the model with memory-efficient settings
        model = xgb.XGBRegressor()
        model.load_model(latest_model_file)

        st.info(
            f"Loaded model from {latest_model_file} and metadata from {latest_metadata_file}."
        )
        return model, metadata
    except MemoryError as me:
        st.error(
            f"MemoryError: Unable to load model due to insufficient memory: {me}. "
            "Try closing other applications, reducing the batch size, or increasing available memory."
        )
        return None, None
    except Exception as e:
        st.error(f"Error loading model or metadata: {e}")
        return None, None


# Prepare test set data (March 31, 2019) with batch prediction
def prepare_test_set_data(feature_df, metadata, model):
    # Filter for test set (March 31, 2019)
    test_df = feature_df[
        (feature_df["hourly_timestamp"] >= "2019-03-31 00:00:00")
        & (feature_df["hourly_timestamp"] < "2019-04-01 00:00:00")
    ].copy()

    # Add predictions by running inference on the test set in batches
    feature_columns = metadata["feature_columns"]
    try:
        # Batch prediction to reduce memory usage
        batch_size = 200  # Reduced batch size to lower memory usage
        predictions = []
        for i in range(0, len(test_df), batch_size):
            batch = test_df.iloc[i : i + batch_size]
            X_batch = batch[feature_columns]
            batch_preds = model.predict(X_batch).round().astype(int)
            predictions.extend(batch_preds)
            # Free memory
            del X_batch
            del batch_preds
            gc.collect()  # Explicit garbage collection after each batch

        test_df["predicted_ride_count"] = predictions
        test_df["actual_ride_count"] = test_df["ride_count"]
        return test_df
    except Exception as e:
        st.error(f"Error generating test set predictions: {e}")
        return None


# Streamlit dashboard
st.title("NYC Taxi Demand Prediction Monitoring Dashboard")

# Load data
feature_df = load_feature_store()
predictions_df = load_predictions()
model, metadata = get_latest_model_and_metadata()

# Check if data loaded successfully
if any(df is None for df in [feature_df, predictions_df, model, metadata]):
    st.stop()

# Prepare test set data
test_df = prepare_test_set_data(feature_df, metadata, model)
if test_df is None:
    st.stop()

# Sidebar: Zone selection
st.sidebar.header("Filter Options")
all_zones = sorted(feature_df["PULocationID"].unique())
selected_zones = st.sidebar.multiselect(
    "Select Zones", all_zones, default=all_zones[:5]
)
if not selected_zones:
    st.sidebar.warning("Please select at least one zone.")
    st.stop()

# Filter data based on selected zones
test_df_filtered = test_df[test_df["PULocationID"].isin(selected_zones)]
predictions_df_filtered = predictions_df[
    predictions_df["PULocationID"].isin(selected_zones)
]

# Section 1: Offline vs. Online Metrics
st.header("Offline vs. Online Error Metrics")
offline_test_mae = metadata["metrics"]["test_mae"]
offline_cv_mae_mean = metadata["cross_validation"]["cv_mae_mean"]

# Compute online MAE for the test set (March 31, 2019)
test_mae_per_hour = (
    test_df_filtered.groupby("hourly_timestamp")
    .apply(
        lambda x: mean_absolute_error(x["actual_ride_count"], x["predicted_ride_count"])
    )
    .reset_index(name="mae")
)

# Plot MAE over time
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=test_mae_per_hour["hourly_timestamp"],
        y=test_mae_per_hour["mae"],
        mode="lines+markers",
        name="Online MAE (Test Set)",
    )
)
fig.add_trace(
    go.Scatter(
        x=[
            test_mae_per_hour["hourly_timestamp"].min(),
            test_mae_per_hour["hourly_timestamp"].max(),
        ],
        y=[offline_test_mae, offline_test_mae],
        mode="lines",
        name=f"Offline Test MAE ({offline_test_mae:.2f})",
        line=dict(dash="dash"),
    )
)
fig.add_trace(
    go.Scatter(
        x=[
            test_mae_per_hour["hourly_timestamp"].min(),
            test_mae_per_hour["hourly_timestamp"].max(),
        ],
        y=[offline_cv_mae_mean, offline_cv_mae_mean],
        mode="lines",
        name=f"Offline CV MAE Mean ({offline_cv_mae_mean:.2f})",
        line=dict(dash="dash"),
    )
)
fig.update_layout(
    title="MAE Over Time: Online vs. Offline Metrics",
    xaxis_title="Timestamp",
    yaxis_title="Mean Absolute Error (MAE)",
    template="plotly_dark",
)
st.plotly_chart(fig)

# Section 2: Predictions vs. Actuals Scatter Plot
st.header("Predictions vs. Actuals")
# Use test set data only
fig = px.scatter(
    test_df_filtered,
    x="actual_ride_count",
    y="predicted_ride_count",
    color="hourly_timestamp",
    hover_data=["PULocationID"],
    title="Predictions vs. Actuals (Test Set: March 31, 2019)",
    labels={
        "actual_ride_count": "Actual Ride Count",
        "predicted_ride_count": "Predicted Ride Count",
    },
    template="plotly_dark",
)
fig.add_trace(
    go.Scatter(
        x=[0, test_df_filtered["actual_ride_count"].max()],
        y=[0, test_df_filtered["actual_ride_count"].max()],
        mode="lines",
        name="Perfect Prediction",
        line=dict(dash="dash", color="white"),
    )
)
st.plotly_chart(fig)

# Section 3: Demand Heatmap
st.header("Demand Heatmap Across Manhattan Zones")
st.info(
    "Showing predicted demand for 2019-04-01 00:00:00. Actual values are unavailable due to missing April 2019 data."
)

# Approximate coordinates for Manhattan zones (placeholder)
zone_coords = {
    4: [40.7231, -73.9821],  # Alphabet City
    12: [40.7046, -74.0137],  # Battery Park City
    13: [40.7150, -74.0155],  # Battery Park
    24: [40.7375, -73.9813],  # Chelsea
    41: [40.8031, -73.9642],  # Central Harlem
    246: [40.7520, -74.0010],  # West Chelsea/Hudson Yards
    249: [40.7349, -74.0050],  # West Village
    261: [40.7033, -74.0170],  # World Trade Center
    262: [40.7769, -73.9539],  # Yorkville
    263: [40.7794, -73.9615],  # Upper West Side
}

# Create a Folium map centered on Manhattan
m = folium.Map(location=[40.7589, -73.9851], zoom_start=12, tiles="CartoDB dark_matter")

# Add predicted demand as a heatmap (using circles for simplicity)
for _, row in predictions_df_filtered.iterrows():
    zone_id = row["PULocationID"]
    if zone_id in zone_coords:
        folium.CircleMarker(
            location=zone_coords[zone_id],
            radius=row["predicted_ride_count"] / 20,  # Scale radius by demand
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.6,
            popup=f"Zone {zone_id}: {row['predicted_ride_count']} rides",
        ).add_to(m)

# Display the map
st_folium(m, width=700, height=500)

# Section 4: Summary Metrics
st.header("Summary Metrics")
col1, col2 = st.columns(2)
col1.metric("Offline Test MAE", f"{offline_test_mae:.2f}")
col2.metric("Online Test MAE (Avg)", f"{test_mae_per_hour['mae'].mean():.2f}")

# Section 5: Model Drift Warning
st.header("Model Drift Detection")
avg_online_mae = test_mae_per_hour["mae"].mean()
if avg_online_mae > offline_test_mae * 1.2:  # 20% threshold
    st.warning(
        "⚠️ Potential model drift detected! Average Online Test MAE is significantly higher than Offline Test MAE."
    )
else:
    st.success("✅ No significant model drift detected.")
