import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Cricket (1).csv", encoding='latin1')

# Drop unwanted column
df.drop('Player', axis=1, inplace=True)

# Split Span column
df[['start', 'end']] = df['Span'].str.split('-', expand=True)

df['start'] = df['start'].astype(int)
df['end'] = df['end'].astype(int)

# Experience column
df['exp'] = df['end'] - df['start']

# Drop Span
df.drop('Span', axis=1, inplace=True)

# Clean HS column
df['HS'] = df['HS'].str.extract(r'(\d+)')
df['HS'] = df['HS'].astype(int)

# Scaling
scaler = MinMaxScaler()

x_scaled = scaler.fit_transform(df)

df_scaled = pd.DataFrame(x_scaled, columns=df.columns)

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)

kmeans.fit(df_scaled)

df_scaled['Label'] = kmeans.labels_

# Features & Target
x = df_scaled.iloc[:, :-1]
y = df_scaled.iloc[:, -1]

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    train_size=0.80,
    random_state=42
)

# Logistic Regression
model = LogisticRegression(random_state=42)

model.fit(x_train, y_train)

# Save files
joblib.dump(model, "cricket_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(x.columns.tolist(), "columns.pkl")

print("Model Saved Successfully!")