🚀 DYNAMIC RESOURCE ALLOCATION AND PREDICTIVE MANAGEMENT SYSTEM FOR REAL-TIME WORKLOAD OPTIMIZATION
🔧 Optimize cloud CPU usage with intelligent, real-time workload predictions using Machine Learning.

🧰 Tech Stack
🐍 Python 3.x

📊 Scikit-learn, Pandas, NumPy

🧠 XGBoost, Random Forest, Linear Regression

🎯 Streamlit (Web UI)

🧱 Joblib (Model Saving)

📈 Matplotlib / Seaborn (Optional Visualization)

🌟 Key Features
📁 Upload workload data via CSV or ✍️ enter manually

✅ Validates workload metrics (CPU, memory, GPU, etc.)

🤖 Predicts optimal CPU Allocatable units

📉 Displays prediction trends with interactive graphs

💾 Download predictions as CSV

🪵 View backend logs for transparency

🧠 How It Works
📤 Input: Provide workload metrics (CPU, memory, status, etc.)

🧪 Transform: Data is preprocessed using a saved ColumnTransformer

🔮 Predict: ML model forecasts how much CPU should be allocated

📊 Visualize: Results shown in line charts and tables

📥 Download: Get predictions in one click

🌐 Web App Preview
📌 Try both input methods – instant feedback and export-ready predictions.


🧪 How to Run Locally
bash
Copy
Edit
git clone https://github.com/your-username/smartcloud-optimizer.git
cd smartcloud-optimizer
pip install -r requirements.txt
streamlit run app.py
📁 Project Structure
bash
Copy
Edit
├── app.py                  # Streamlit dashboard
├── artifacts/              # Saved model & preprocessor
│   ├── model.pkl
│   └── preprocessor.pkl
├── assets/                 # App screenshots & icons
├── requirements.txt
└── README.md
📸 Screenshots
CSV Upload	Manual Input	Trend Visualization

📄 License
📝 This project is licensed under the MIT License.


