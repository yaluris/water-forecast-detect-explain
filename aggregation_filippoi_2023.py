import pandas as pd

# 1. Load and preprocess data
df = pd.read_excel('data/data_filippoi_2023.xlsx', sheet_name='DATA2023')
df.columns = ['date', 'time', 'flow1', 'total_flow', 'hourly_flow', 'daily_flow', 'monthly_flow']
# Combine 'date' and 'time' into a single datetime column
df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), errors='coerce')
df.set_index('datetime', inplace=True)
df.drop(['date', 'time', 'flow1'], axis=1, inplace=True)
print(df.head())

# 2. Create aggregated features
daily_df = df.resample('D').agg({
    'hourly_flow': ['sum', 'mean', 'max', 'min', 'std']
})
daily_df.columns = ['_'.join(col).strip() for col in daily_df.columns.values]
print(daily_df.head())

# 3. Extract time features
daily_df['day_of_week'] = daily_df.index.dayofweek
daily_df['day_of_month'] = daily_df.index.day
daily_df['month'] = daily_df.index.month
daily_df['week_of_year'] = daily_df.index.isocalendar().week
daily_df['is_weekend'] = daily_df.index.dayofweek >= 5
print(daily_df.head())

# 4. Calculate rolling statistics and lag features
# Compute the average daily flow over the last 7 days
daily_df['7d_avg_daily_flow'] = daily_df['hourly_flow_sum'].rolling(window=7).mean()
# Compute the standard deviation of daily flow over the last 7 days
daily_df['7d_std_daily_flow'] = daily_df['hourly_flow_sum'].rolling(window=7).std()
# Create lag features (flow values from previous days)
for lag in [1, 2, 3, 7]:
    daily_df[f'lag_{lag}d_flow'] = daily_df['hourly_flow_sum'].shift(lag)
print(daily_df.head(10))

# 5. Save results
daily_df.dropna().to_csv('aggregate_data.csv', index=True)
