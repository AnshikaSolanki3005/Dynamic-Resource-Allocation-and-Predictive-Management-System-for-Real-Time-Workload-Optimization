import streamlit as st
import pandas as pd
import os
from src.utils import load_object
from src.components.data_transformation import DataTransformation

# Initialize transformer
data_transformer = DataTransformation()

def load_model():
    model_path = os.path.join("artifacts", "model.pkl")
    model_bundle = load_object(model_path)
    model = model_bundle["model"]
    expected_features = list(model_bundle["expected_features"])
    return model, expected_features

def predict(input_data, from_manual=False):
    model, expected_features = load_model()  # Load the model and expected features

    # If the input is from manual entry, add dummy values for missing categorical columns
    if from_manual:
        for col in DataTransformation.categorical_columns:
            if col not in input_data.columns:
                input_data[col] = "dummy"

    # Reorder input columns to expected order
    input_data = input_data[DataTransformation.numerical_columns + DataTransformation.categorical_columns]

    # Apply the data transformation pipeline
    transformed_data = data_transformer.transform_uploaded_csv(input_data)

    # Align features exactly as the model expects
    transformed_data = transformed_data.reindex(columns=expected_features, fill_value=0)

    # Make prediction
    prediction = model.predict(transformed_data)
    return prediction

# Streamlit UI
st.title("Dynamic Resource Allocation Optimization")
st.write("Upload a CSV file or manually enter data to predict resource utilization.")

uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df.head())

    if st.button("Predict from CSV"):
        try:
            predictions = predict(df)
            df["Predicted_Resource_Usage"] = predictions
            st.write("Prediction Results:", df)
            st.download_button("Download Predictions", df.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Manual input
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

    try:
        prediction = predict(manual_data, from_manual=True)
        st.write(f"Predicted Resource Usage: {prediction[0]}")
    except Exception as e:
        st.error(f"Error during manual prediction: {e}")
