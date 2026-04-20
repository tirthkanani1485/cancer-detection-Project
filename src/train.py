# src/train.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("../data/data.csv")

# Clean data
if 'id' in df.columns:
    df = df.drop('id', axis=1)

if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Features
X = df[['radius_mean', 'texture_mean', 'area_mean']]
Y = df['diagnosis']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, Y_train)

# Save
joblib.dump(model, "../model/cancer_model.pkl")
joblib.dump(scaler, "../model/scaler.pkl")

print("✅ Model trained and saved successfully!")