💡 DYNAMIC RESOURCE ALLOCATION AND PREDICTIVE MANAGEMENT SYSTEM FOR REAL-TIME WORKLOAD OPTIMIZATION
Optimize Workloads, Maximize Efficiency Instantly

🚀 Overview
This project presents a machine learning-based system to predict CPU allocatable resources in real-time cloud environments. It dynamically manages and optimizes workloads using predictive modeling and intelligent recommendations.

📋 Table of Contents
Overview

Tech Stack

Features

Model Workflow

Web App Preview

How to Run

Directory Structure

Screenshots

License

🛠 Tech Stack
Python

Streamlit

Scikit-learn

Pandas / NumPy
Matplotlib / Seaborn

Joblib

VS Code / GitHub or Github Codespaces

✨ Features
CSV upload & manual input for predictions.

Data validation with error handling.

Predicts CPU allocatable based on 6 workload metrics.

Line chart visualization of prediction trends.

Downloadable output file with predictions.

Web app built using Streamlit (lightweight and interactive).

Logs viewer and model metadata.

🔁 Model Workflow
Data Input (CSV or manual).

Preprocessing with trained ColumnTransformer.

Prediction using a trained ML regression model.

Visualization of trends via charts.

Export Results as CSV.

🌐 Web App Preview

Upload CSV or enter manually to get intelligent CPU allocatable predictions.

⚙️ How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/your-repo.git
cd your-repo
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the app:

bash
Copy
Edit
streamlit run app.py
📁 Directory Structure
bash
Copy
Edit
project/
├── app.py                      # Streamlit web app
├── artifacts/
│   ├── model.pkl               # Trained model and expected features
│   ├── preprocessor.pkl        # ColumnTransformer
│   └── logs.log                # Log file
├── assets/
│   └── ui_preview.png          # Screenshot of web UI
├── requirements.txt
└── README.md
🖼 Screenshots
📥 CSV Upload Section:

🧮 Manual Input & Prediction:

📊 Prediction Graph:

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.
