import os  
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        timestamp: str,
        node: str,
        cpu_workloads: float,
        memory_workloads: float,
        nvidia_gpu_workloads: str,
        status: str,
        condition: str,
        scenario_workloads: int,
        uid: str,
        cpu_allocatable: float, 
        nvidia_gpu_allocatable: float,
        scenario_allocatable: str
    ):
        self.timestamp = timestamp
        self.node = node
        self.cpu_workloads = cpu_workloads
        self.memory_workloads = memory_workloads
        self.nvidia_gpu_workloads = nvidia_gpu_workloads
        self.status = status
        self.condition = condition
        self.scenario_workloads = scenario_workloads
        self.uid = uid
        self.cpu_allocatable = cpu_allocatable
        self.nvidia_gpu_allocatable = nvidia_gpu_allocatable
        self.scenario_allocatable = scenario_allocatable

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'timestamp': [self.timestamp],
                'node': [self.node],
                'cpu_workloads': [self.cpu_workloads],
                'memory_workloads': [self.memory_workloads],
                'nvidia_gpu_workloads': [self.nvidia_gpu_workloads],
                'status': [self.status],
                'condition': [self.condition],
                'scenario_workloads': [self.scenario_workloads],
                'uid': [self.uid],
                'cpu_allocatable': [self.cpu_allocatable],
                'nvidia_gpu_allocatable': [self.nvidia_gpu_allocatable],
                'scenario_allocatable': [self.scenario_allocatable],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)