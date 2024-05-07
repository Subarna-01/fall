import numpy as np
from scipy.stats import kurtosis, skew
import joblib

model = joblib.load('fall_detection.pkl')

def predict_fall(accel_data, gyro_data, model):
    # Extracting required features from the provided accelerometer and gyroscope data
    acc_max_4th_sec = max(accel_data[:10])
    lin_max_4th_sec = max(accel_data[:10])
    acc_kurtosis_val = kurtosis(accel_data)
    acc_skewness_val = skew(accel_data)
    gyro_kurtosis_val = kurtosis(gyro_data) if gyro_data else 0  # Check if gyro_data is empty
    gyro_skewness_val = skew(gyro_data) if gyro_data else 0  # Check if gyro_data is empty
    post_gyro_max = max(gyro_data[50:]) if gyro_data and len(gyro_data) > 50 else 0  # Check if gyro_data has enough elements
    post_lin_max = max(accel_data[50:]) if len(accel_data) > 50 else 0  # Check if accel_data has enough elements
    
    # Prepare feature vector for prediction
    feature_vector = np.array([acc_max_4th_sec, gyro_kurtosis_val, lin_max_4th_sec, acc_skewness_val, gyro_skewness_val, post_gyro_max, post_lin_max])
    
    # Perform fall detection using the trained machine learning model
    prediction = model.predict(feature_vector.reshape(1, -1))
    
    # Return prediction result
    return "Fall" if prediction == 1 else "No fall"