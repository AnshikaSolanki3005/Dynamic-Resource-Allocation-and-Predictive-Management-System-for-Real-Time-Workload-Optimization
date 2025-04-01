import os  
import sys
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from src.exception import CustomException
from src.utils import load_object

# Initialize FastAPI app
app = FastAPI()

# Define request model
class ResourceInput(BaseModel):
    timestamp: str
    node: str
    cpu_workloads: float
    memory_workloads: float
    nvidia_gpu_workloads: str
    status: str
    condition: str
    scenario_workloads: int
    uid: str
    cpu_allocatable: float
    nvidia_gpu_allocatable: float
    scenario_allocatable: str

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, features):
        try:
            model = load_object(file_path=self.model_path)
            preprocessor = load_object(file_path=self.preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, data: ResourceInput):
        self.data = data

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'timestamp': [self.data.timestamp],
                'node': [self.data.node],
                'cpu_workloads': [self.data.cpu_workloads],
                'memory_workloads': [self.data.memory_workloads],
                'nvidia_gpu_workloads': [self.data.nvidia_gpu_workloads],
                'status': [self.data.status],
                'condition': [self.data.condition],
                'scenario_workloads': [self.data.scenario_workloads],
                'uid': [self.data.uid],
                'cpu_allocatable': [self.data.cpu_allocatable],
                'nvidia_gpu_allocatable': [self.data.nvidia_gpu_allocatable],
                'scenario_allocatable': [self.data.scenario_allocatable]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

# Define FastAPI route
@app.post("/predict")
def predict_resource_allocation(input_data: ResourceInput):
    custom_data = CustomData(input_data)
    df = custom_data.get_data_as_data_frame()

    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)

    return {"recommended_memory_allocatable": prediction.tolist()}
