import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model and preprocessor
model_bundle = joblib.load("artifacts/model.pkl")
model = model_bundle["model"]
expected_features = model_bundle["expected_features"]
preprocessor = joblib.load("artifacts/preprocessor.pkl")

st.set_page_config(page_title="SmartCloud Optimizer", layout="wide")
st.title("📊 SmartCloud Optimizer Dashboard")

st.markdown("""
This app predicts **CPU Allocatable** based on workload metrics and suggests smart allocation.
You can upload workload data or manually input metrics.
""")

# Helper function to make predictions
def predict_cpu_allocatable(input_df):
    transformed = preprocessor.transform(input_df)
    return model.predict(transformed)

# Sidebar input method
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose Input Method", ["Upload CSV", "Manual Input"])

# ======================================
# 📁 Upload CSV Section
# ======================================
if input_method == "Upload CSV":
    file = st.file_uploader("Upload your workload data CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.subheader("📃 Uploaded Data")
        st.dataframe(df.head())

        # ✅ Validate required columns
        required_cols = [
            'cpu_workloads', 'memory_workloads', 'nvidia_com_gpu_workloads',
            'scenario_workloads', 'status', 'condition'
        ]
        missing = [col for col in required_cols if col not in df.columns]

        if missing:
            st.error(f"Uploaded file is missing required columns: {missing}")
        else:
            try:
                predictions = predict_cpu_allocatable(df)
                df["Predicted_CPU_Allocatable"] = predictions

                st.subheader("🔮 Predictions")
                st.dataframe(df)

                st.subheader("📈 Prediction Trend")

                # Add a slider to choose how many rows to visualize
                limit = st.slider("Select number of rows to visualize", min_value=100, max_value=min(len(df), 10000), step=100, value=500)

                # Plot only the first 'limit' rows
                st.line_chart(df["Predicted_CPU_Allocatable"].head(limit))


                csv = df.to_csv(index=False)
                st.download_button("Download Predictions", data=csv, file_name="predicted_allocations.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ======================================
# 🧮 Manual Input Section
# ======================================
else:
    st.subheader(":pencil2: Manual Input")
    cpu = st.number_input("CPU Workloads", 0.0, 100.0, 30.0)
    mem = st.number_input("Memory Workloads", 0.0, 100.0, 45.0)
    gpu = st.number_input("GPU Workloads", 0.0, 100.0, 20.0)
    scenario = st.number_input("Scenario Workloads", 0.0, 100.0, 10.0)
    status = st.selectbox("Status", ["Running", "Pending", "Failed"])
    condition = st.selectbox("Condition", ["Normal", "Degraded", "Critical"])

    manual_df = pd.DataFrame([{
        "cpu_workloads": cpu,
        "memory_workloads": mem,
        "nvidia_com_gpu_workloads": gpu,
        "scenario_workloads": scenario,
        "status": status,
        "condition": condition
    }])

    if st.button("Predict", key="predict_manual"):
        try:
            prediction = predict_cpu_allocatable(manual_df)[0]
            st.success(f"Predicted CPU Allocatable: {prediction:.2f} units")
        except ValueError as ve:
            st.error(f"Preprocessing Error: {ve}")
        except Exception as e:
            st.error(f"Unexpected Error: {e}")

# ======================================
# 🔍 Model Info Section
# ======================================
with st.expander("🔧 Model Info"):
    st.write("**Model Type:**", type(model).__name__)
    st.write("**Feature Count:**", len(expected_features))
    st.code("\n".join(expected_features))

# ======================================
# 📄 Logs Viewer (optional)
# ======================================
try:
    with open("artifacts/logs.log", "r") as f:
        logs = f.read()
    with st.expander("📝 View Logs"):
        st.text_area("Logs", logs, height=300)
except:
    pass
