import os
import sys
sys.path.insert(0, os.path.abspath("C:/Users/ASUS/Desktop/Project/src/components"))
from src.components.data_transformation import DataTransformation

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Starting data ingestion...")
        try:
            # Read the raw dataset
            df = pd.read_csv('notebook/data.csv')
            logging.info("Dataset loaded successfully")

            # Ensure artifact directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)

            # Train-test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Data ingestion completed successfully")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()

    # Data transformation
    transformation_obj = DataTransformation()
    train_arr, test_arr= transformation_obj.initiate_data_transformation(train_data_path, test_data_path)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))