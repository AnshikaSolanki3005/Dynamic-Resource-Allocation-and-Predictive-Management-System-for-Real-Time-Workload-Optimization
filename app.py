from flask import Flask,request,render_template
import pickle
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.utils import save_object
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application=Flask(__name__)

app=application

# route for a home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_resource_allocation():
    if request.method=='GET':
        return render_template('home.html')

    else:
        data=CustomData(
            timestamp=request.form.get('timestamp'),
            node=request.form.get('node'),
            cpu_workloads=float(request.form.get('cpu_workloads')),
            memory_workloads=float(request.form.get('memory_workloads')),
            nvidia_com_gpu_workloads=float(request.form.get('nvidia_com_gpu_workloads')),
            status=request.form.get('status'),
            condition=request.form.get('condition'),
            scenario_workloads=request.form.get('scenario_workloads'),
            uid=request.form.get('uid'),
            cpu_allocatable=float(request.form.get('cpu_allocatable')),
            nvidia_com_gpu_allocatable=float(request.form.get('nvidia_com_gpu_allocatable')),
            scenario_allocatable=request.form.get('scenario_allocatable')
        )

        pred_df=data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline=PredictPipeline()
        results=predict_pipeline.predict(pred_df)
        return render_template('home.html',results=results[0])

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)