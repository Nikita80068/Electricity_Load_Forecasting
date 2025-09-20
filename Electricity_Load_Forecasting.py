#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout




# In[5]:


df = pd.read_csv('powerdemand_5min_2021_to_2024_with weather.csv', parse_dates=['datetime'], index_col='datetime')

df.head()


# In[6]:


df = df.resample('h').mean()  # If you want hourly data instead of 5-min

# Fill missing values
df = df.interpolate(method='time')

# Optional: feature engineering
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month

# Suppose dataset has load column called 'demand' and weather like 'temperature', 'humidity'
features = ['Power demand', 'temp', 'rhum', 'hour', 'day', 'month']
data = df[features]

# Scaling
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# In[7]:


def create_sequences(data, target_idx, window_size=24):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, :])
        y.append(data[i, target_idx])
    return np.array(X), np.array(y)

window_size = 24  # using past 24 hours to predict next
target_col = features.index('Power demand')

X, y = create_sequences(data_scaled, target_idx=target_col, window_size=window_size)


# In[8]:


split_ratio = 0.8
split = int(len(X) * split_ratio)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)


# In[9]:


model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


# In[10]:


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    shuffle=False
)


# In[11]:


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print("Test MSE:", mse)
print("Test MAE:", mae)


# In[12]:


plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Delhi Electricity Demand: Actual vs Predicted')
plt.show()


# In[13]:


plt.figure(figsize=(14,6))

# Short-Term (first 100 hours for clarity)
plt.subplot(2,1,1)
plt.plot(y_test[:100].flatten(), label="Actual")
plt.plot(y_pred[:100].flatten(), label="Predicted")
plt.title("Short-Term Forecast (Next 24 Hours)")
plt.legend()


# In[14]:


# Long-Term (first 500 hours for clarity)
plt.subplot(2,1,2)
plt.plot(y_test[:500].flatten(), label="Actual")
plt.plot(y_pred[:500].flatten(), label="Predicted")
plt.title("Long-Term Forecast (Next 7 Days)")
plt.legend()

plt.tight_layout()
plt.show()


# In[15]:


# Peak Load Prediction
daily_peak = df['Power demand'].resample('D').max()

plt.figure(figsize=(12,5))
plt.plot(daily_peak, label="Daily Peak Load", color="red")
plt.title("Peak Load Prediction (Daily Max Demand)")
plt.xlabel("Date")
plt.ylabel("Load (MW)")
plt.legend()
plt.show()


# In[16]:


# Load vs Temperature
plt.figure(figsize=(12,5))
plt.scatter(df['temp'], df['Power demand'], alpha=0.5, c='blue')
plt.title("Electricity Demand vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Demand (MW)")
plt.show()


# In[17]:


# Load vs Humidity
plt.figure(figsize=(12,5))
plt.scatter(df['rhum'], df['Power demand'], alpha=0.5, c='green')
plt.title("Electricity Demand vs Humidity")
plt.xlabel("Humidity (%)")
plt.ylabel("Demand (MW)")
plt.show()


# In[18]:


df.keys()


# In[19]:


plt.figure(figsize=(12,6))
plt.stackplot(df.index, df['Power demand'], df['dwpt'], df['wdir'],
              labels=['Power Demand','dwpt','wdir'], alpha=0.8)
plt.title("Electricity Demand with Renewable Integration")
plt.xlabel("Date")
plt.ylabel("MW")
plt.legend(loc="upper left")
plt.show()


# In[20]:


# Simple anomaly detection using Z-score
demand_mean, demand_std = df['Power demand'].mean(), df['Power demand'].std()
z_scores = (df['Power demand'] - demand_mean) / demand_std
anomalies = df[np.abs(z_scores) > 3]  # points beyond 3σ

plt.figure(figsize=(12,5))
plt.plot(df.index, df['Power demand'], label="Power Demand")
plt.scatter(anomalies.index, anomalies['Power demand'], color="red", label="Anomalies")
plt.title("Anomaly Detection in Electricity Demand")
plt.xlabel("Date")
plt.ylabel("Load (MW)")
plt.legend()
plt.show()


# In[21]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Define demand categories
def categorize_load(value):
    if value < 0.33:
        return "Low"
    elif value < 0.66:
        return "Medium"
    else:
        return "High"

# Scale back predictions to 0-1 if needed (assuming you already scaled with MinMaxScaler)
y_test = [categorize_load(v) for v in y_test]
y_pred = [categorize_load(v) for v in y_pred]

# Step 2: Build confusion matrix
labels = ["Low", "Medium", "High"]
cm = confusion_matrix(y_test, y_pred, labels=labels)

# Step 3: Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Electricity Load Categories")
plt.show()


# In[22]:


import seaborn as sns

# Compute correlation
corr = df[['Power demand', 'temp', 'rhum', 'hour', 'dayofweek', 'month']].corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix - Electricity Demand & Features")
plt.show()


# In[23]:


df.describe().T.style.background_gradient(cmap = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True))


# In[24]:


df.isna().sum()


# In[25]:


categorical_val = []
continous_val = []
for column in df.columns:
    print('==============================')
    print(f"{column} : {df[column].unique()}")
    if len(df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[28]:


get_ipython().system('jupyter nbconvert --to script Electricity_Load_Forecasting.ipynb')




# In[ ]:


get_ipython().system('streamlit run Electricity_Load_Forecasting.py')


# In[ ]:




