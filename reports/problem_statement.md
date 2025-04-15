# Problem Statement

We need to create a predictive model to forecast the number of yellow taxi rides that will happen in Manhattan (New York City) with data from 2019 (January, February and March). We'll complement our work with additional Borough information (Taxi Zone Lookup Table) and NYC weather datasets for feature engineering.

- Predict taxi demand in NYC in the next 60 minutes.
- Model taxi demand across all Manhattan zones (e.g. Zone 113 "Lower Manhattan).

[TLC Trip Record Data Link](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

## **ML Project Steps**

### **Step 1: Transform raw data into (features, targets) and train-test split.**
This step involves preparing the raw data for modeling by transforming it into a format suitable for supervised machine learning. Here’s how we’ll approach the sub-steps:

- **Validate the Raw Data and Data Cleaning**: We’ll start by loading the TLC Trip Record Data for yellow taxis (January to March 2019) in Parquet format in `data_ingestor.py` script. We’ll use Dask to handle the large dataset efficiently. Validation will include:

- Checking for missing values in critical columns like pickup/dropoff timestamps, locations (PULocationID, DOLocationID), and passenger counts.
- Ensuring timestamps are in the correct format and within the expected range (Jan-Mar 2019).
- Filtering out invalid records (e.g., negative trip distances, missing location IDs, or trips outside Manhattan using the Taxi Zone Lookup Table).
- Handling outliers, such as unusually high passenger counts or trip distances, using statistical methods like IQR or domain knowledge (e.g., max reasonable taxi ride distance in Manhattan). The cleaned data will be saved to data/interim/.

- **Feature engineering**: In `feature_engineering.py` script, we’ll create features to capture patterns in taxi demand. Features will include:

- Temporal Features: Hour of day, day of week, and whether it’s a holiday or weekend.
- Historical Ride Counts: Lagged features, such as the number of rides in the previous 1, 2, or 24 hours for each zone.
- Weather Data: Temperature, precipitation, and possibly wind speed from the NYC weather dataset, merged with the trip data based on timestamps.
- Zone Characteristics: Using the Taxi Zone Lookup Table, we’ll add features like zone area, population density (if available), or categorical features like “Lower Manhattan” for zone 113. We’ll merge the weather and borough data with the trip data using timestamps and location IDs.

- **Aggregate Raw Data into Time-Series**: In `make_dataset.py` script, we’ll aggregate the data into hourly counts of rides per Manhattan zone. The target will be the number of rides in the next hour for each zone. For example, for Zone 113 at 2022-06-01 12:00, the target is the number of rides from 13:00 to 14:00, and features will include historical counts, weather, and zone info up to 12:00.

- **Transform Time-Series into (Features, Target)**: We’ll create a supervised learning dataset where each row represents a zone and timestamp, with columns for features (e.g., lagged ride counts, weather) and the target (next hour’s ride count). This aligns with the approach “From raw data to training data,” where raw data is processed into (features, target) pairs.

- **Train-Test Split**: We’ll perform a time-based split: January and February 2019 for training, and March 2019 for testing. This ensures the model is evaluated on future data, mimicking real-world forecasting.

### **Step 2: Build a Baseline Model**
We’ll implement a simple baseline model where the predicted demand for the next hour is the demand from the last hour. For example, if Zone 113 had 50 rides from 12:00 to 13:00, the baseline prediction for 13:00 to 14:00 is 50 rides. We’ll calculate the Mean Absolute Error (MAE) on the test set (March 2019) to evaluate the baseline.

### **Step 3: Improve the Baseline Using ML**
We’ll train a machine learning model to outperform the baseline. The four improvement strategies will be:

- **Increase Training Data**: We’ll ensure we’re using all available data from January and February 2019, potentially adding more historical lags (e.g., previous 48 hours).
- **Increase Features**: Add more features like interaction terms (e.g., temperature * precipitation), or additional weather variables (e.g., humidity).
- **Try Another ML Algorithm**: Start with a model like LightGBM (good for tabular data and time-series), then experiment with XGBoost or a simple neural network.
- **Fine-Tune Hyperparameters**: Use MLflow to track experiments and tune hyperparameters like learning rate, tree depth, or number of estimators for LightGBM/XGBoost.

We’ll train the model in `train_model.py` script and generate predictions in `predict_model.py` script, saving models to the models/ directory. MLflow will log metrics (e.g., MAE) and parameters for each run.

### **Step 4: Put Our Model to Work (Batch-Scoring System)**
We’ll implement a batch-scoring system with three pipelines, following these three steps:

- **Feature Pipeline**: Processes raw data into features (e.g., aggregates hourly counts, merges weather data).
- **Training Pipeline**: Trains the model periodically (e.g., daily or weekly) using the latest data.
- **Inference Pipeline**: Uses the trained model to make predictions on new data.

These pipelines will interact with a Feature Store/Model Registry (e.g., using MLflow’s model registry) to store features and models. We’ll implement this in models/batch-scoring_system folder, ensuring modularity.

In a production environment, the system should be set up to automatically ingest and process new data as it arrives (e.g., via an API, streaming service, or scheduled batch process). The system should not require manual downloading of future data (e.g., April 2019 data). Instead, it should be designed to ingest and process new data seamlessly as it arrives in a production environment. Expect a scalable setup with predictions ready for monitoring.

Since we’re in a simulated environment, we’ll use the most recent data from features_2019_march.parquet (up to 2019-03-31 23:00:00) and predict for 2019-04-01 00:00:00.

### **Step 5: Build a Monitoring Dashboard**
Using Streamlit, we’ll create a dashboard (`frontend.py`) to monitor model performance, as shown in the “And monitoring dashboard” visual. The dashboard will:

- Fetch model predictions and actual targets from the test set (March 2019).
- Plot online error metrics (e.g., MAE over time) and compare them to offline metrics from training.
- Visualize predictions vs. actuals, possibly with a heatmap of demand across Manhattan zones (similar to the Streamlit screenshot).

The dashboard will help us detect model drift or performance degradation in a production-like setting.
