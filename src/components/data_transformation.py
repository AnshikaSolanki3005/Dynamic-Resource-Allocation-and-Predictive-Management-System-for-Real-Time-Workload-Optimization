import os
import sys
from dataclasses import dataclass

import numpy as numpy
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging


class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numerical_columns=["cpu_workloads", "memory_workloads", "nvidia_com_gpu_workloads", "cpu_allocatable", "memory_allocatable", "nvidia_com_gpu_allocatable"]
            categorical = ['timestamp','node', 'status', 'condition', 'scenario_workloads', 'scenario_allocatable', 'uid']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("label_encoder", label_encoder()),
                    ("scaler",StandardScaler())
                ]
            )

            logging.info("Numerical columns encoding completed")

            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline,categorical)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read and train data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="memory_allocatable"
            numerical_columns=[]
        
        except:
            pass