import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """Creates and returns the preprocessing pipeline for numerical and categorical features."""
        numerical_columns = [
            "cpu_workloads", "memory_workloads", "nvidia_com_gpu_workloads",
            "cpu_allocatable", "nvidia_com_gpu_allocatable"
        ]
        categorical_columns = [
            'timestamp', 'node', 'status', 'condition', 'scenario_workloads', 'scenario_allocatable', 'uid'
        ]

        # Pipelines
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Column Transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        """Transforms the train and test datasets using the preprocessing pipeline."""
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        preprocessor = self.get_data_transformer_object()

        # Apply transformations
        train_arr = preprocessor.fit_transform(train_df)
        test_arr = preprocessor.transform(test_df)

        # Save preprocessing object
        save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)
        
        return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path