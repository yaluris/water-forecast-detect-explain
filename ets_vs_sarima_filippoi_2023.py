# pip install openpyxl pandas matplotlib statsmodels scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# 1. Load and clean data
df = pd.read_excel('C:/Users/georg/Desktop/Thesis/data_filippoi_2023.xlsx', sheet_name='DATA2023')
df.columns = ['date', 'time', 'flow1', 'total_flow', 'hourly_flow', 'daily_flow', 'monthly_flow']
# Combine 'date' and 'time' into a single datetime column
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
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

# 2. Train-test split (last 7 days = test)
train = df.iloc[:-24*7]
test = df.iloc[-24*7:]

# 3. Holt-Winters (ETS)
# Additive trend, additive seasonality (seasonal fluctuations are constant in magnitude), daily seasonality (24 hours cycle)
hw_model = ExponentialSmoothing(
    train['hourly_flow'],
    trend='add',
    seasonal='add',
    seasonal_periods=24
).fit()
# Generate forecast for the test period
hw_forecast = hw_model.forecast(len(test))

# 4. SARIMA
# Maybe use ACF/PACF plots or auto_arima to find optimal parameters
sarima_model = SARIMAX(
    train['hourly_flow'],
    order=(1,1,1),  # (AR, differencing, MA)
    seasonal_order=(1,1,1,24),  # (Seasonal AR, differencing, MA, period)
    enforce_stationarity=False,
    enforce_invertibility=False
).fit()
sarima_forecast = sarima_model.forecast(steps=len(test))

# 5. Evaluation
def evaluate(y_true, y_pred, label):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label} - MAE: {mae:.3f}, RMSE: {rmse:.3f}")

evaluate(test['hourly_flow'], hw_forecast, "Holt-Winters (ETS)")
evaluate(test['hourly_flow'], sarima_forecast, "SARIMA")

# 6. Plotting 
plt.figure(figsize=(16, 6))
plt.plot(test.index, test['hourly_flow'], label='Actual', color='black')
plt.plot(test.index, hw_forecast, label='Holt-Winters (ETS)', linestyle='--')
plt.plot(test.index, sarima_forecast, label='SARIMA', linestyle='--')
plt.title("Forecast Comparison: Holt-Winters vs SARIMA")
plt.xlabel("Time")
plt.ylabel("Hourly Water Consumption (m³/hour)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Residuals Analysis
# Calculate residuals (ensure indices match)
hw_residuals = test['hourly_flow'] - hw_forecast.values
sarima_residuals = test['hourly_flow'] - sarima_forecast.values
plt.figure(figsize=(15, 5))
plt.plot(test.index, hw_residuals, label='Holt-Winters Residuals', alpha=0.7)
plt.plot(test.index, sarima_residuals, label='SARIMA Residuals', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residuals Comparison')
plt.xlabel('Time')
plt.ylabel('Residuals (Actual - Forecast)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
