import numpy as np
from fastapi import FastAPI
from schemas import SensorData
from predict import predict_fall,model

app = FastAPI()

@app.post('/predict')
async def predict(sensor_data: SensorData):
    combined_acc_data = []
    combined_gyro_data = []
    
    for x, y, z in zip(sensor_data.acc_x, sensor_data.acc_y, sensor_data.acc_z):
        combined_acc_data.append(np.sqrt(x**2 + y**2 + z**2))

    for x, y, z in zip(sensor_data.gyro_x, sensor_data.gyro_y, sensor_data.gyro_z):
        combined_gyro_data.append(np.sqrt(x**2 + y**2 + z**2))

    result = predict_fall(combined_acc_data, combined_gyro_data, model)
    return { 'res': result}
    