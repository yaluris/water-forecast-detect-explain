# pip install pandas openpyxl xgboost scikit-learn matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Load and clean data
df = pd.read_excel('data/data_filippoi_2024.xlsx', sheet_name='DATA2024')
df.columns = ['datetime', 'flow', 'flow_calibration', 'total_flow', 'hourly_flow', 'daily_flow', 'monthly_flow']
df['datetime'] = pd.to_datetime(df['datetime'].astype(str), errors='coerce')
df.set_index('datetime', inplace=True)
df['hourly_flow'] = pd.to_numeric(df['hourly_flow'], errors='coerce')
# Keep only 'hourly_flow' column (with datetime index) and drop rows with missing values
df = df[['hourly_flow']].dropna()
print(df.head())
# Plot the hourly flow time series
plt.figure(figsize=(15, 5))
plt.plot(df.index, df['hourly_flow'], label='Hourly Water Consumption')
plt.title('Hourly Water Consumption Over Time')
plt.xlabel('Time')
plt.ylabel('m³/hour')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Feature engineering
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
# Add lag features (previous values)
for lag in range(1, 25):  # Previous 24 hours
    df[f'lag_{lag}'] = df['hourly_flow'].shift(lag)
# Drop rows with NaN values caused by lagging
df.dropna(inplace=True)
# Define features and target
X = df.drop('hourly_flow', axis=1)
y = df['hourly_flow']
# Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.06, shuffle=False
)

# 3. Train model
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
model.fit(X_train, y_train)

# 4. Recursive Multi-Step Forecasting (no data leakage)
recursive_preds = []
history = df.iloc[:len(X_train)].copy()  # Only the training portion
forecast_dates = X_test.index
for current_time in forecast_dates:
    # Get lag features from most recent known values in history
    lag_values = [history['hourly_flow'].iloc[-lag] for lag in range(1, 25)]
    lag_dict = {f'lag_{i+1}': lag_values[i] for i in range(24)}
    # Add time-based features
    lag_dict['hour'] = current_time.hour
    lag_dict['dayofweek'] = current_time.dayofweek
    lag_dict['month'] = current_time.month
    # Predict
    features_df = pd.DataFrame([lag_dict])[X_train.columns]
    prediction = model.predict(features_df)[0]
    recursive_preds.append(prediction)
    # Append prediction to history to simulate future context
    new_row = pd.DataFrame({'hourly_flow': [prediction]}, index=[current_time])
    history = pd.concat([history, new_row])
# Convert predictions to Series
y_pred_recursive = pd.Series(recursive_preds, index=forecast_dates)

# 5. Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred_recursive))
mae = mean_absolute_error(y_test, y_pred_recursive)
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# 6. Plot actual vs predicted values
plt.figure(figsize=(15, 5))
plt.plot(y_test.index, y_test.values, label='Actual', color='black')
plt.plot(y_test.index, y_pred_recursive, label='Predicted', linestyle='--', color='orange')
plt.title("XGBoost Recursive Forecast")
plt.xlabel("Time")
plt.ylabel("Hourly Water Consumption (m³/hour)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

