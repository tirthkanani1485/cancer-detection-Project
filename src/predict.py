# src/predict.py

import joblib
import numpy as np

# Load
import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "model", "cancer_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def predict(radius, texture, area):
    data = np.array([[radius, texture, area]])
    scaled = scaler.transform(data)
    result = model.predict(scaled)[0]
    
    return "Cancer Detected ⚠️" if result == 1 else "No Cancer ✅"


# Test
if __name__ == "__main__":
    print(predict(17.99, 10.38, 1001.0))
