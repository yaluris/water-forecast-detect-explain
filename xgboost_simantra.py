# pip install pandas openpyxl xgboost scikit-learn matplotlib requests

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import requests

# 1. Load and clean data
df = pd.read_excel("data/data_simantra_metrics.xlsx")
# Convert timestamp
df['datetime'] = pd.to_datetime(df['date'], unit='s')
df = df.sort_values(['deviceid', 'datetime'])
# Calculate hourly consumption (diff from cumulative)
df['hourly_consumption'] = df.groupby('deviceid')['value'].diff().fillna(0)
# Drop rows with negative consumption
df = df[df['hourly_consumption'] >= 0]
# Keep only 'deviceid', 'datetime', 'hourly_consumption' columns
df = df[['deviceid', 'datetime', 'hourly_consumption']]
# Add latitude and longitude
unique_coords = pd.read_excel("data/data_simantra_meters.xlsx", usecols=['deviceid', 'latitude', 'longitude'])
df = df.merge(unique_coords, on='deviceid', how='left')
# print(df.head())

# 2. Feature engineering
# Temporal features
df['hour'] = df['datetime'].dt.hour
df['day_of_week'] = df['datetime'].dt.dayofweek
df['day'] = df['datetime'].dt.day
# holiday
# Lag features
for lag in range(1, 25):  # Last 24 hours
    df[f'lag_{lag}'] = df.groupby('deviceid')['hourly_consumption'].shift(lag)
# Rolling mean features
df['rolling_mean_6'] = df.groupby('deviceid')['hourly_consumption'].transform(lambda x: x.shift(1).rolling(window=6).mean())
df.dropna(inplace=True)
# Per-device statistical features
device_stats = df.groupby('deviceid')['hourly_consumption'].agg(['mean', 'std', 'max', 'min', 'median']).reset_index()
device_stats.columns = ['deviceid', 'dev_mean', 'dev_std', 'dev_max', 'dev_min', 'dev_median']
df = df.merge(device_stats, on='deviceid', how='left')
# Spatial features
kmeans = KMeans(n_clusters=10, random_state=42)  # KMeans clustering
unique_coords['location_cluster'] = kmeans.fit_predict(unique_coords[['latitude', 'longitude']])
df = df.merge(unique_coords[['deviceid', 'location_cluster']], on='deviceid', how='left')
df = pd.get_dummies(df, columns=['location_cluster'])  # One-hot encode cluster
# Weather features (add weather features via APIs like Open-Meteo based on latitude/longitude)
latitude, longitude = 40.345876, 23.309706
start_date = df['datetime'].min().date().isoformat()
end_date = df['datetime'].max().date().isoformat()
weather_url = (
    f"https://archive-api.open-meteo.com/v1/archive?"
    f"latitude={latitude}&longitude={longitude}"
    f"&hourly=temperature_2m,precipitation"
    f"&start_date={start_date}&end_date={end_date}&timezone=auto"
)
response = requests.get(weather_url)
weather_data = []
if response.status_code == 200:
    data = response.json()
    for i, time_str in enumerate(data['hourly']['time']):
        weather_data.append({
            'datetime': pd.to_datetime(time_str),
            'temperature_2m': data['hourly']['temperature_2m'][i],
            'precipitation': data['hourly']['precipitation'][i],
        })
else:
    raise Exception(f"Weather API failed: {response.status_code}")
weather_df = pd.DataFrame(weather_data)
df = pd.merge(df, weather_df, on='datetime', how='left')
# print(df.head())
# Define features and target
features = ['hour', 'day_of_week', 'day'] + \
           [f'lag_{lag}' for lag in range(1, 25)] + ['rolling_mean_6'] + \
           ['dev_mean', 'dev_std', 'dev_max', 'dev_min', 'dev_median'] + \
           [col for col in df.columns if col.startswith('location_cluster_')] + \
           ['temperature_2m', 'precipitation']
target = 'hourly_consumption'

# Does the use of spatial features make sense in this case?????

# 3. Split into training and testing
train_dfs = []
test_dfs = []
holdout_hours = 24  # Forecast horizon (24 hours)
for device_id, group in df.groupby('deviceid'):
    group = group.sort_values('datetime')
    if len(group) > holdout_hours:
        train_dfs.append(group.iloc[:-holdout_hours])
        test_dfs.append(group.iloc[-holdout_hours:])
train_df = pd.concat(train_dfs)
test_df = pd.concat(test_dfs)
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

# 4. Train model
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)
model.fit(X_train, y_train)

# 5. Predict and evaluate (compute RMSE per device)
y_pred = model.predict(X_test)
test_df['prediction'] = y_pred
rmse_per_device = test_df.groupby('deviceid').apply(
    lambda x: np.sqrt(mean_squared_error(x['hourly_consumption'], x['prediction']))
).reset_index(name='rmse')
print(rmse_per_device.head(20))

