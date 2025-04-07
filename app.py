import streamlit as st
import pandas as pd
import numpy as np
import os
from src.utils import load_object
from src.components.data_transformation import DataTransformation

def load_model():
    """Load the trained model and preprocessor."""
    model_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
    model = load_object(model_path)
    preprocessor = load_object(preprocessor_path)
    return model, preprocessor

def predict(input_data):
    """Make predictions using the trained model."""
    model, preprocessor = load_model()
    transformed_data = preprocessor.transform(input_data)
    prediction = model.predict(transformed_data)
    return prediction

# Streamlit UI
st.title("Dynamic Resource Allocation Optimization")
st.write("Upload a CSV file or manually enter data to predict resource utilization.")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data:", df.head())
    
    if st.button("Predict"):
        predictions = predict(df)
        df["Predicted_Resource_Usage"] = predictions
        st.write("Prediction Results:", df)
        st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")

st.subheader("Or enter data manually:")
cpu_workloads = st.number_input("CPU Workloads", min_value=0.0, value=50.0)
memory_workloads = st.number_input("Memory Workloads", min_value=0.0, value=16.0)
nvidia_com_gpu_workloads = st.number_input("GPU Workloads", min_value=0.0, value=5.0)
cpu_allocatable = st.number_input("CPU Allocatable", min_value=0.0, value=100.0)
nvidia_com_gpu_allocatable = st.number_input("GPU Allocatable", min_value=0.0, value=10.0)

if st.button("Predict Manually"):
    manual_data = pd.DataFrame({
        "cpu_workloads": [cpu_workloads],
        "memory_workloads": [memory_workloads],
        "nvidia_com_gpu_workloads": [nvidia_com_gpu_workloads],
        "cpu_allocatable": [cpu_allocatable],
        "nvidia_com_gpu_allocatable": [nvidia_com_gpu_allocatable]
    })
    manual_prediction = predict(manual_data)
    st.write(f"Predicted Resource Usage: {manual_prediction[0]}")
