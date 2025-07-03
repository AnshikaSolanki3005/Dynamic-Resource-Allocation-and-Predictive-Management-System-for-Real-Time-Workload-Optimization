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
from src.exception import CustomException
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            # Adjusted based on typical workload structure
            numerical_columns = [
                "cpu_workloads", "memory_workloads", "nvidia_com_gpu_workloads",
                "scenario_workloads"
            ]

            categorical_columns = [
                "status", "condition"
            ]

            # Pipelines
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))  # Compatible with sklearn >=1.2
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            logging.info("Preprocessor object created successfully")
            return preprocessor

        except Exception as e:
            raise CustomException(str(e), sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data")

            # ✅ Define the correct target column
            target_column = "cpu_allocatable"

            # ✅ Check if all required columns are present
            required_columns = [
                "cpu_workloads", "memory_workloads", "nvidia_com_gpu_workloads",
                "scenario_workloads", "status", "condition", target_column
            ]
            missing_cols = [col for col in required_columns if col not in train_df.columns]
            if missing_cols:
                raise CustomException(f"Missing columns in training data: {missing_cols}", sys)

            # ✅ Split X and y
            input_features_train = train_df.drop(columns=[target_column])
            target_feature_train = train_df[target_column]

            input_features_test = test_df.drop(columns=[target_column])
            target_feature_test = test_df[target_column]

            # ✅ Get and apply preprocessor
            preprocessor = self.get_data_transformer_object()
            input_features_train_transformed = preprocessor.fit_transform(input_features_train)
            input_features_test_transformed = preprocessor.transform(input_features_test)

            logging.info("Transformation applied successfully")

            # ✅ Save the preprocessor
            save_object(self.data_transformation_config.preprocessor_obj_file_path, preprocessor)

            # ✅ Combine features and targets
            train_array = np.c_[input_features_train_transformed, target_feature_train.to_numpy()]
            test_array = np.c_[input_features_test_transformed, target_feature_test.to_numpy()]

            return train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(str(e), sys)
