import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object, load_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
 
    numerical_columns = [
        "cpu_workloads", "memory_workloads", "nvidia_com_gpu_workloads",
        "cpu_allocatable", "nvidia_com_gpu_allocatable"
    ]
    categorical_columns = [
        'timestamp', 'node', 'status', 'condition',
        'scenario_workloads', 'scenario_allocatable', 'uid'
    ]

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # Numerical features pipeline
        num_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        # Categorical features pipeline
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Combine the two pipelines
        preprocessor = ColumnTransformer(
            transformers=[
                ("num_pipeline", num_pipeline, self.numerical_columns),
                
                ("cat_pipeline", cat_pipeline, self.categorical_columns)
            ]
        )

        return preprocessor

    def initiate_data_transformation(self, train_path, test_path):
        # Load the training and testing data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Get the preprocessing transformer
        preprocessor = self.get_data_transformer_object()

        # Apply transformations to the train and test datasets
        train_arr = preprocessor.fit_transform(train_df)
        test_arr = preprocessor.transform(test_df)

        # Save the preprocessor
        save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

        return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

    def transform_uploaded_csv(self, input_df):
        try:
            # Load the preprocessor object
            preprocessor = load_object(self.data_transformation_config.preprocessor_obj_file_path)

            # Ensure the input data has the correct columns and order
            expected_columns = self.numerical_columns + self.categorical_columns
            input_df = input_df[expected_columns]

            # Transform the input data using the preprocessor
            transformed_array = preprocessor.transform(input_df)

            # Get the feature names (output from OneHotEncoder)
            feature_names = preprocessor.get_feature_names_out()

            # Convert the transformed array to a DataFrame with the proper columns
            transformed_df = pd.DataFrame(transformed_array, columns=feature_names)

            return transformed_df

        except Exception as e:
            raise Exception(f"Transformation failed: {e}")
