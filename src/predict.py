# src/predict.py

import joblib
import numpy as np

# Load
model = joblib.load("../model/cancer_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

def predict(radius, texture, area):
    data = np.array([[radius, texture, area]])
    scaled = scaler.transform(data)
    result = model.predict(scaled)[0]
    
    return "Cancer Detected ⚠️" if result == 1 else "No Cancer ✅"


# Test
if __name__ == "__main__":
    print(predict(17.99, 10.38, 1001.0))