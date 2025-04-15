# NYC Taxi Demand Forecasting

Predict hourly yellow taxi demand across Manhattan zones using NYC trip data, weather conditions, and time-based features. This project integrates data engineering, machine learning, and dashboarding to simulate a real-world demand forecasting system with batch inference and monitoring capabilities.

---

## 🚀 Project Objective

Forecast the number of yellow taxi rides in Manhattan (New York City) for the next 60 minutes using historical trip records, borough zone data, and NYC weather. This prediction system will:

- Anticipate hourly demand per pickup zone (e.g., Zone 113: Lower Manhattan)
- Enable operational optimization for taxi dispatch systems
- Simulate a batch production-ready ML deployment with real-time extensions

---

## 🗃️ Data Sources

| Source Type  | Description                                                                                     |
|--------------|-------------------------------------------------------------------------------------------------|
| [TLC Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | Yellow taxi trips from Jan to Mar 2019 (Parquet format)                           |
| Taxi Zone Lookup Table   | CSV mapping `PULocationID` to NYC boroughs and zones (e.g., Upper East Side)        |
| NYC Weather Data         | Hourly temperature, precipitation, humidity, and wind speed for time-series modeling|

---

## 🧠 Machine Learning Workflow

The project follows the classical ML pipeline structure: ingest → feature engineering → modeling → batch scoring → dashboarding.

### 🔹 Step 1: Data Ingestion & Preparation

- Implemented in `data_ingestor.py` and `make_dataset.py`
- Uses Dask for large-scale data loading and cleaning
- Filters out invalid or non-Manhattan records using Taxi Zone Lookup
- Cleans and prepares time-aligned trip records

### 🔹 Step 2: Feature Engineering

- Implemented in `feature_engineering.py`
- Feature types:
  - **Temporal**: Hour of day, day of week, is_weekend, is_holiday
  - **Lagged Rides**: Ride counts at t-1, t-2, ..., t-24 hours
  - **Weather**: Temperature, wind, precipitation
  - **Zone Metadata**: Categorical labels for borough/zones (from Lookup table)

### 🔹 Step 3: Baseline Model

- Implemented in `train_baseline_model.py`
- Simple persistence model: predicts next-hour ride count = last-hour ride count
- Serves as a benchmark for more advanced models

### 🔹 Step 4: Advanced ML Models

- Implemented in `train_advanced_models.py` using LightGBM and XGBoost
- Strategies:
  - Feature expansion: interaction terms, extended lags, richer weather features
  - Algorithm experimentation: LightGBM, XGBoost, Neural Networks
  - Hyperparameter tuning: tree depth, learning rate, boosting rounds
- Uses MLflow for tracking experiments and metrics

### 🔹 Step 5: Batch Inference System

- Located in `models/batch-scoring_system/`
- Modular 3-stage pipeline:
  1. `00_data_ingestor.py`: loads latest raw data
  2. `01_feature_pipeline.py`: builds features from new data
  3. `02_training_pipeline.py` + `03_inference_pipeline.py`: trains model & forecasts
- Generates predictions for `2019-04-01 00:00:00` (simulating next-hour inference)

### 🔹 Step 6: Monitoring Dashboard

- Streamlit dashboard implemented in `dashboard.py`
- Features:
  - MAE trends over time
  - Actual vs. Predicted demand visualizations
  - Manhattan heatmap of ride volume
  - Drift analysis and zone-level prediction errors

---

## 🧾 Folder Structure

```
nyc-taxi-demand-forecasting/
├── data/
│   ├── external/       # Weather and zone lookup
│   ├── raw/            # Raw trip data (Parquet)
│   ├── interim/        # Cleaned but unaggregated
│   └── processed/      # Aggregated time-series data
├── docs/               # Project documentation (Sphinx)
├── models/
│   ├── batch-scoring_system/
│   │   ├── 00_data_ingestor.py
│   │   ├── 01_feature_pipeline.py
│   │   ├── 02_training_pipeline.py
│   │   ├── 03_inference_pipeline.py
│   │   ├── feature_store/
│   │   ├── models_and_metadata/
│   │   └── predictions/
│   ├── dashboard.py
│   ├── predict_model.py
│   └── train_baseline_model.py
├── notebooks/          # Exploratory notebooks
├── references/
│   ├── data_dictionary_trip_records_yellow.pdf
│   └── TLC_Taxi_Zones_Manhattan.jpg
├── reports/
│   ├── problem_statement.md
│   ├── lookup_table.md
│   ├── large_dataset_assessment.md
│   └── figures/
├── src/
│   ├── data/
│   │   ├── data_ingestor.py
│   │   └── make_dataset.py
│   ├── features/
│   │   └── feature_engineering.py
│   ├── models/
│   │   ├── train_advanced_models.py
│   │   └── predict_model.py
│   └── visualization/
│       ├── analyze_model.py
│       └── analyze_busy_zones.py
├── environment.yml     # Conda environment specification
├── LICENSE
└── README.md
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/SebastianGarrido2790/nyc-taxi-demand-forecasting.git
cd nyc-taxi-demand-forecasting
```

2. **Create the environment**
```bash
conda env create -f environment.yml
conda activate taxi_demand_prediction
```

3. **Download data**
- Place raw trip data (Parquet) in `data/raw/`
- Place NYC weather data in `data/external/`
- Place `taxi+_zone_lookup.csv` in `data/external/`

4. **Run pipelines**
```bash
# Data ingestion
python src/data/data_ingestor.py

# Feature engineering
python src/features/feature_engineering.py

# Train baseline
python src/models/train_baseline_model.py

# Train advanced models
python src/models/train_advanced_models.py

# Run batch prediction
python models/batch-scoring_system/03_inference_pipeline.py

# Launch dashboard
streamlit run models/dashboard.py
```

---

## 🧪 Evaluation Metrics

| Metric          | Description                                                   |
|-----------------|---------------------------------------------------------------|
| MAE             | Measures average error in predictions                         |
| RMSE            | Penalizes large errors more than MAE                          |
| Zone MAE        | Evaluates performance per Manhattan zone                      |
| MAE Drift       | Tracks model drift using online metrics vs. offline validation|

---

## 📊 Dashboard Preview

- **MAE over Time**: Monitor prediction accuracy during March 2019
- **Zone Heatmap**: Visualize actual/predicted demand across NYC zones
- **Prediction vs. Ground Truth**: Spot high-error scenarios
- **Drift Detection**: Compare online and offline MAE

---

## 📌 Future Work

- Implement real-time ingestion (Kafka, APIs)
- Add zone-level metadata (population density, venues)
- Explore Deep Learning (RNNs, Transformers)
- Integrate anomaly detection for outlier demand

---

## 📜 License

This project is licensed under the MIT License. See the [LICENCE](./LICENCE.txt)) file for details.

---

## 🙌 Acknowledgments

- NYC TLC for public trip data
- National Weather Service API
- Taxi Zone Lookup data from NYC Open Data
