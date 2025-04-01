from fastapi import FastAPI
from pydantic import BaseModel
from src.pipeline.predict_pipeline import PredictPipeline, CustomData
import uvicorn

# Initialize the FastAPI app
app = FastAPI()

# Define a Pydantic model for incoming requests
class InputData(BaseModel):
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

# Create an instance of the PredictPipeline class
predict_pipeline = PredictPipeline()

@app.post("/predict")
def predict(data: InputData):
    # Convert the received data to a CustomData object
    custom_data = CustomData(
        timestamp=data.timestamp,
        node=data.node,
        cpu_workloads=data.cpu_workloads,
        memory_workloads=data.memory_workloads,
        nvidia_gpu_workloads=data.nvidia_gpu_workloads,
        status=data.status,
        condition=data.condition,
        scenario_workloads=data.scenario_workloads,
        uid=data.uid,
        cpu_allocatable=data.cpu_allocatable,
        nvidia_gpu_allocatable=data.nvidia_gpu_allocatable,
        scenario_allocatable=data.scenario_allocatable,
    )

    # Get the data as a DataFrame for prediction
    df = custom_data.get_data_as_data_frame()

    # Make predictions using the PredictPipeline
    predictions = predict_pipeline.predict(df)

    # Return the prediction as a response
    return {"predictions": predictions.tolist()}

# If running this file directly, start the Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
