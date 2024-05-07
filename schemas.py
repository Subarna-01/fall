from pydantic import BaseModel

class SensorData(BaseModel):
    acc_x: list[float]
    acc_y: list[float]
    acc_z: list[float]

    gyro_x: list[float]
    gyro_y: list[float]
    gyro_z: list[float]