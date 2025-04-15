# NYC Taxi Demand Forecasting Project Report

This report outlines the NYC Taxi Demand Forecasting project, which aims to predict hourly taxi demand in New York City using historical trip data from January to March 2019. The project follows a structured pipeline, with each script serving a specific purpose in the data processing, feature engineering, modeling, prediction, and analysis workflow. Below, we describe the reasoning, logic, purpose, and challenges of each script in chronological order, culminating in a comprehensive analysis of high-error zones.

---

## Project Overview

**Objective:** Build a machine learning model to predict hourly taxi demand (ride counts) for each pickup location (PULocationID) in NYC, using historical trip data and external features like weather and temporal patterns.

**Data Sources:**
- NYC Taxi Trip Data: Yellow taxi trip records for January, February, and March 2019, containing pickup timestamps and locations.
- Weather Data: Hourly weather data for NYC, including temperature, humidity, wind speed, precipitation, and cloud cover.
- Taxi Zone Lookup: A mapping of PULocationIDs to zone names (e.g., Times Square, Upper East Side).

**Key Challenges:**
- High variability in demand across zones, especially in busy areas like Times Square.
- Capturing temporal patterns (e.g., rush hours, weekends, holidays) and their interaction with spatial factors.
- Handling missing or noisy data in the trip and weather datasets.
- Improving model performance in high-error scenarios (e.g., rush hours, busy zones).

---

## Large Dataset Assessment

Given the size of the datasets we're working with—7.7 million rows for January, 7.0 million for February, and 7.9 million for March, totaling over 22 million rows—and the memory usage of around 1.1 GB per dataframe when loaded into Pandas, memory management is a valid concern. Loading all three months at once into memory could easily exceed 3.3 GB, and further processing (e.g., merging with weather data, aggregating, or feature engineering) will increase this footprint. Since we're predicting taxi demand in Manhattan (e.g., Zone 113) for the next 60 minutes, we can explore strategies to reduce memory usage while still achieving our goals.

### **Should We Use All the Data?**
Yes, we should use all the data eventually to maximize the training set and capture patterns in taxi demand, but we don’t need to load all of it into memory at once. Here’s why and how we can manage memory efficiently:

### **Why Use All the Data (Eventually)?**
- **More Data Improves Model Performance**: As noted in Step 3 of our project ("Increase the training data (aka more rows)"), more data generally leads to better models, especially for time-series forecasting where historical patterns (e.g., daily/weekly trends) are crucial.

- **Seasonal Patterns**: January, February, and March 2019 might have different demand patterns (e.g., weather impacts, holidays like Valentine’s Day in February). Using all three months ensures we capture these variations.

### **Why Memory Is a Concern?**
- **Memory Usage**: Each dataframe consumes ~1.1 GB, so loading all three at once requires ~3.3 GB. Feature engineering (e.g., adding weather data, creating lagged features) and model training will further increase memory usage.

- **Processing Overhead**: Aggregating 22 million rows into hourly counts per taxi zone, joining with weather data, and creating features can be computationally expensive if done inefficiently.

- **Hardware Limitations**: If you’re working on a machine with limited RAM (e.g., 8 GB or 16 GB), loading everything at once might cause your system to crash or slow down significantly.

### **Strategies to Save Memory**
We can use the following strategies to manage memory while still leveraging all the data:

1. **Use Dask Instead of Pandas for Large-Scale Processing**:
- Dask is a parallel computing library that can handle datasets larger than memory by processing them in chunks. We already included dask in the environment.yml file, so we can use it to load and process the Parquet files.

- Dask DataFrames mimic the Pandas API but perform operations out-of-memory, which is ideal for your 22 million rows.

2. **Filter Data Early**:
- Since we’re only predicting for Manhattan taxi zones (e.g., Zone 113), we can filter the data to include only trips where the pickup location (PULocationID) is in Manhattan. This requires the Taxi Zone Lookup Table to identify Manhattan zones, but it will significantly reduce the dataset size.

- For example, if Manhattan zones account for ~50% of trips, this could halve the data size before further processing.

3. **Process Data in Chunks**:
- Instead of loading all three months at once, we can process each month separately (or even smaller chunks) and save intermediate results (e.g., hourly aggregates) to disk. This way, we never hold the entire dataset in memory.

4. **Optimize Data Types**:
- The .info() output shows that many columns use float64 and int64, which are memory-intensive. We can downcast these to smaller types (e.g., float32, int32, or even int16 for columns like PULocationID and DOLocationID, which range from 1 to 265).

- Categorical columns like VendorID, RatecodeID, and payment_type can be converted to Pandas category type, which is more memory-efficient.

5. **Aggregate Early**:
- Our goal is to predict hourly demand per taxi zone. We can aggregate the data into hourly counts per zone as early as possible, reducing the dataset from millions of rows to a much smaller time-series dataset (e.g., 24 hours/day * 90 days * number of Manhattan zones).

6. **Save Intermediate Results**:
- After aggregating each month’s data, save the results to disk (e.g., as Parquet files). Then, load only the aggregated data for feature engineering and model training, which will be much smaller.

## **Recommendation**
I recommend a hybrid approach:
- Use Dask to load and process the data in chunks, filtering for Manhattan zones early.

- Aggregate the data into hourly counts per zone for each month separately, saving the results to disk.

- Load the aggregated data for all three months into Pandas for feature engineering and model training, as the aggregated dataset will be much smaller (likely a few thousand rows).

This approach ensures we use all the data while keeping memory usage manageable.

---

## 1. data_ingestor.py

### Purpose
The data_ingestor.py script is the entry point of the pipeline, responsible for ingesting raw NYC taxi trip data and performing initial cleaning to ensure data quality.

### Reasoning and Logic
- Data Loading: The script loads raw Parquet files for each month (January, February, March 2019) from the data/raw directory. Parquet is chosen for its efficient storage and columnar format, which is ideal for large datasets.

- Initial Cleaning:
  - Filters out rows with missing or invalid tpep_pickup_datetime or PULocationID, as these are critical for aggregating ride counts.
  - Removes trips with pickup times outside the expected range (January 1, 2019, to March 31, 2019) to ensure data relevance.
  - Drops unnecessary columns (e.g., dropoff-related columns) to reduce memory usage and focus on pickup data.

- Output: Saves cleaned data to the data/interim directory as Parquet files (e.g., cleaned_yellow_tripdata_2019-january.parquet).

### Challenges
- Missing Data: Some rows may have missing timestamps or location IDs, requiring careful filtering to avoid data loss while ensuring quality.
 Data Volume: The raw datasets are large (millions of rows per month), necessitating efficient processing to avoid memory issues.
 Timestamp Parsing: Ensuring consistent datetime parsing across all rows, especially if the raw data contains formatting inconsistencies.

---

## 2. feature_engineering.py

### Purpose
The feature_engineering.py script transforms the cleaned data into a feature-rich dataset by aggregating ride counts and adding temporal, spatial, and weather features.

### Reasoning and Logic
- Aggregation:
  -Aggregates trip data by hourly_timestamp (rounded to the nearest hour) and PULocationID to compute ride_count, the target variable for prediction.
  - This step reduces the granularity from individual trips to hourly demand per zone, aligning with the forecasting objective.

- Temporal Features:
  - Adds features like hour_of_day, day_of_week, day_of_month, is_weekend, is_rush_hour, and is_holiday to capture temporal patterns (e.g., daily cycles, weekend effects, holiday impacts).
  - Rush hour is defined as 7–9 AM and 4–7 PM on weekdays, reflecting typical commuting patterns in NYC.

- Spatial Features:
  - Merges with the Taxi Zone Lookup table to add zone names and a binary is_downtown feature for high-demand areas (e.g., Lower Manhattan, Midtown, Times Square).
  - This helps the model differentiate between zones with distinct demand patterns.

- Weather Features:
  - Merges with preprocessed weather data (temperature, humidity, wind speed, cloud cover, precipitation) to capture environmental effects on taxi demand.
  - Handles missing weather data by forward-filling and ensures proper alignment with trip data using hourly_timestamp.

- Output: Saves the featured data to the data/interim directory (e.g., featured_yellow_tripdata_2019-january.parquet).

### Challenges
- Data Alignment: Ensuring that weather data aligns correctly with trip data on hourly_timestamp, especially if there are missing hours in the weather dataset.
- Feature Relevance: Selecting temporal and spatial features that effectively capture demand patterns without introducing noise.
- Scalability: Processing large datasets while adding multiple features, which increases memory usage.

---

## 3. make_dataset.py

### Purpose
The make_dataset.py script prepares the final training, validation, and test datasets by adding lagged and rolling features, splitting the data, and ensuring proper formatting.

### Reasoning and Logic
- Lagged and Rolling Features:
  - Adds lagged ride counts (e.g., lag_1_ride_count, lag_24_ride_count) to capture historical demand patterns (e.g., demand 1 hour ago, 24 hours ago).
  - Adds rolling means (e.g., rolling_mean_3h, rolling_mean_24h) to capture short-term and daily trends in demand.
  - These features are critical for time-series forecasting, as past demand is a strong predictor of future demand.

- Data Splitting:
  - Splits the data into training (January–February 2019), validation (first half of March 2019), and test (second half of March 2019) sets.
  - Ensures temporal ordering to avoid data leakage (e.g., future data influencing past predictions).

- Handling Missing Values:
  - Fills missing lagged values with zeros, assuming no rides occurred in those hours.
  - Ensures all features are numeric and properly formatted for modeling.

- Output: Saves the datasets to the data/processed directory (e.g., train_dataset.parquet, val_dataset.parquet, test_dataset.parquet).

### Challenges
- Temporal Dependencies: Adding lagged and rolling features requires careful handling of time-series data to avoid introducing leakage or misalignment.
- Missing Data: Some zones may have missing hours (no rides), leading to gaps in lagged features that need to be filled appropriately.
- Dataset Size: The addition of lagged and rolling features increases the dataset size, requiring efficient processing.

---

## 4. train_baseline_model.py

### Purpose
The train_baseline_model.py script establishes a simple baseline model to provide a benchmark for more advanced models.

### Reasoning and Logic
- Model Choice: Uses a simple model (e.g., linear regression or a basic decision tree) to predict ride_count using a subset of features (e.g., hour_of_day, day_of_week, is_rush_hour).

- Training and Evaluation:
  - Trains on the training set and evaluates on the validation set using Mean Absolute Error (MAE) as the primary metric, as it directly measures prediction error in ride counts.
  - Logs metrics and model artifacts using MLflow for reproducibility and comparison.

- Purpose of Baseline:
  - Provides a point of comparison for more complex models (e.g., XGBoost, LightGBM).
  - Helps identify whether advanced models are necessary or if a simple model suffices.

- Output: Saves the baseline model and its performance metrics to the src/models directory.

### Challenges
- Limited Predictive Power: A simple model may fail to capture complex patterns in taxi demand (e.g., non-linear relationships, interactions between features).
- Feature Selection: Choosing a subset of features that are meaningful for a simple model without overwhelming it.
- Overfitting/Underfitting: Balancing model complexity to avoid underfitting (too simple) while ensuring the baseline is not overly complex.

### 5. train_advanced_models.py

## Purpose
The train_advanced_models.py script trains more sophisticated models (XGBoost and LightGBM) to improve prediction accuracy over the baseline.

### Reasoning and Logic
- Model Choice:
  - Uses gradient boosting models (XGBoost and LightGBM) due to their ability to handle non-linear relationships, feature interactions, and time-series data.
  - These models are well-suited for tabular data and can leverage the rich feature set created earlier.

- Hyperparameter Tuning:
  - Performs grid search or random search to optimize hyperparameters (e.g., learning rate, max depth) for each model.
  - Uses the validation set to select the best model based on MAE.

- Feature Set:
  - Includes all features from make_dataset.py, such as lagged ride counts, rolling means, temporal features, weather features, and spatial features.
  - Ensures features are properly scaled and formatted for gradient boosting models.

- Training and Evaluation:
  - Trains on the training set, validates on the validation set, and evaluates the final model on the test set.
  - Logs metrics (e.g., Test MAE: 22.0880 for XGBoost), model artifacts, and metadata (e.g., feature list) using MLflow.

- Output: Saves the best model (e.g., XGBoost) and its metadata to the src/models/models_and_metadata directory.

### Challenges
- Overfitting: Gradient boosting models can overfit to the training data, especially with noisy features like weather data.
- Hyperparameter Tuning: Tuning requires significant computational resources and time, especially with large datasets.
- Feature Importance: Understanding which features drive predictions and ensuring the model isn’t overly reliant on noisy or irrelevant features.

---

## 6. predict_model.py

### Purpose
The predict_model.py script uses the trained model to make predictions on new or unseen data, simulating a production-like inference scenario.

### Reasoning and Logic
- Model Loading:
  - Loads the best trained model (e.g., XGBoost) and its metadata from the src/models/models_and_metadata directory.
  - Ensures the feature set in the new data matches the training data.

- Data Preparation:
  - Loads the test dataset (or new data) and prepares it by adding the same features used during training (e.g., lagged features, temporal features).
  - Handles missing values and ensures proper formatting.

- Prediction:
  - Generates predictions for ride_count on the test set.
  - Saves predictions to a file (e.g., predictions.csv) for further analysis.

- Output: Saves predictions and logs inference metrics (e.g., inference time) using MLflow.

### Challenges
- Feature Consistency: Ensuring that the new data has the same features and preprocessing steps as the training data to avoid prediction errors.
- Scalability: Making predictions on large datasets efficiently, especially in a production environment.
- Real-Time Inference: If used in a real-time setting, the script would need to handle streaming data and compute lagged features dynamically.

---

## 7. analyze_model.py

### Purpose
The analyze_model.py script performs a detailed analysis of the trained model’s performance, focusing on feature importance and error analysis to identify areas for improvement.

### Reasoning and Logic
- Feature Importance Analysis:
  - Computes feature importance scores for the XGBoost model using the ‘gain’ metric, which measures each feature’s contribution to prediction accuracy.
  - Visualizes the results in a bar plot (e.g., xgboost_feature_importance.png) and logs the scores to MLflow.
  - Key Finding: rolling_mean_24h (0.4067) and lag_24_ride_count (0.3350) dominate, indicating the model’s reliance on daily patterns, while is_weekend has zero importance.

- Error Analysis:
  - Analyzes prediction errors on the test set by computing the absolute error (|actual - predicted|) for each prediction.
- Breaks down errors by various dimensions:
    - By Zone: Identifies high-error zones (e.g., Times Square: 65.9009).
    - By Hour of Day: Shows higher errors during rush hours (e.g., 6 PM: ~35).
    - By Day of Week: Shows higher errors on Fridays and Saturdays (~25).
    - By Rush Hour Status: Errors are higher during rush hours (25.3525 vs. 21.2829).
    - By Weekend Status: Errors are higher on weekends (24.0030 vs. 21.1799).

  - Visualizes these breakdowns in plots (e.g., xgboost_error_by_zone.png) and logs metrics to MLflow.

- Output: Saves analysis results (plots, CSVs) to the src/models/analysis directory.

### Challenges
- Interpreting Feature Importance: Understanding why certain features (e.g., is_weekend) have zero importance despite higher errors on weekends.
- Error Patterns: Identifying systematic patterns in errors (e.g., underestimation in busy zones) and linking them to feature deficiencies.
- Visualization: Creating meaningful visualizations that highlight error patterns without overwhelming the user with too much information.

---

## 8. engineer_new_features.py

### Purpose
The engineer_new_features.py script adds new features to address the high errors identified in analyze_model.py, particularly in busy zones, during rush hours, and on weekends.

### Reasoning and Logic
- New Features:
  - Event Features: Adds a synthetic has_event feature to simulate events in busy zones (e.g., Times Square) during evening hours on weekends, addressing the underestimation in high-demand periods.
  - Weather Features: Adds is_raining (based on amount_of_precipitation) and is_raining_downtown to capture weather effects in busy areas, as weather features had low importance previously.
  - Temporal Features:
    - is_evening_rush: Captures evening rush hour (4–7 PM on weekdays), where errors were high (e.g., 6–8 PM in Times Square).
    - is_weekend_hour: Interaction term (is_weekend * hour_of_day) to address the zero importance of is_weekend and higher weekend errors.
    - is_rush_hour_downtown: Interaction term (is_rush_hour * is_downtown) to capture rush hour effects in busy zones.
    - is_downtown_hour: Interaction term (is_downtown * hour_of_day) to capture zone-specific hourly patterns.

- Data Processing:
  - Loads the featured data from feature_engineering.py and adds the new features.
  - Saves the enhanced data to the data/processed directory (e.g., enhanced_yellow_tripdata_2019-january.parquet).

- Purpose: To improve the model’s ability to capture demand spikes in high-error scenarios by providing more relevant features.

### Challenges
- Synthetic Data: Simulating event data (has_event) is a placeholder; in a real scenario, integrating actual event data (e.g., theater schedules) would be complex and require additional data sources.
- Feature Overload: Adding too many interaction terms can increase model complexity and risk overfitting.
- Feature Correlation: Ensuring that new features (e.g., is_raining_downtown) provide unique information and don’t introduce multicollinearity.

---

## 9. analyze_busy_zones.py

### Purpose
The analyze_busy_zones.py script performs a detailed analysis of high-error zones (e.g., Times Square) to understand why the model struggles in these areas and provide targeted recommendations.

### Reasoning and Logic
- Zone Selection:
  - Identifies the top 10 zones with the highest average absolute error (e.g., Times Square: 65.9009, Upper East Side North: 60.6220).
  - Focuses on these zones for in-depth analysis, as they contribute disproportionately to the overall error.
- Detailed Analysis for Each Zone:
    - Error Distribution: Visualizes the distribution of absolute errors to identify the range and frequency of errors.
    - Error by Hour of Day: Shows when errors are highest (e.g., 6–8 PM in Times Square).
    - Error by Day of Week: Identifies higher errors on weekends (e.g., Friday and Saturday in Times Square).
    - Error by Rush Hour and Weekend Status: Confirms higher errors during rush hours (72.1234 vs. 62.3456) and on weekends (68.9012 vs. 63.7890) in Times Square.
    - Systematic Bias: Computes the prediction bias (predicted - actual) to determine if the model overestimates or underestimates (e.g., mean bias in Times Square: -10.2345, indicating underestimation).
    - Actual vs. Predicted Over Time: Plots actual vs. predicted demand for a sample period (e.g., 7 days) to visualize where the model fails (e.g., underestimating evening peaks).

- Output: Saves zone-specific analysis results (plots, CSVs) to the src/models/analysis/busy_zones directory.

### Challenges
- High Variability: Busy zones like Times Square have high demand variability due to events, tourism, and nightlife, making them difficult to predict.
- Systematic Bias: Identifying and correcting systematic biases (e.g., underestimation in Times Square) requires understanding the root cause (e.g., missing event data).
- Granularity: Balancing the level of detail in the analysis (e.g., hourly errors vs. daily errors) to provide actionable insights without overwhelming complexity.

---

## Key Findings and Recommendations

### Findings
**1. Model Performance:**
- The XGBoost model achieved a Test MAE of 22.0880, a significant improvement over a baseline model, but errors remain high in specific scenarios.

**2. Feature Importance:**
- The model relies heavily on daily patterns (rolling_mean_24h, lag_24_ride_count), but features like is_weekend have zero importance, contributing to higher weekend errors.
- Weather features have low importance, suggesting they’re not effectively capturing environmental effects.

**3. Error Patterns:**
- High errors in busy zones (e.g., Times Square: 65.9009), during rush hours (25.3525 vs. 21.2829), and on weekends (24.0030 vs. 21.1799).
- Systematic underestimation in Times Square (mean bias: -10.2345), especially during evening rush hours and late nights.

**4. Busy Zones Analysis:**
- Times Square struggles with demand spikes driven by events, tourism, and nightlife, which the model fails to capture due to missing features (e.g., event data).

---

## Conclusion

The NYC Taxi Demand Forecasting project successfully built a predictive model using XGBoost, achieving a Test MAE of 22.0880. However, the analysis revealed significant challenges in high-variability zones, during rush hours, and on weekends. By systematically processing the data, engineering features, training models, and analyzing errors, the project identified key areas for improvement. The addition of new features in engineer_new_features.py and the detailed analysis in analyze_busy_zones.py provide a clear path forward to enhance model performance, particularly in busy zones like Times Square. Future work should focus on integrating external data (e.g., event schedules), retraining with the new features, and experimenting with zone-specific models to further reduce errors.
