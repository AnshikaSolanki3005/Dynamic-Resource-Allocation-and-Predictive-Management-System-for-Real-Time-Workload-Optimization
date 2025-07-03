import os
import sys
from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models, load_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor()
            }

            params = {
                "XGBRegressor": {
                    'learning_rate': [0.1, 0.01, 0.05, 0.001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {}
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, params=params
            )

            # 🔍 Print scores safely (diagnostics only)
            for model_name, score in model_report.items():
                logging.info(f"Model '{model_name}' R² score: {score:.4f}")

            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

           # if best_model_score < 0.3:
             #   raise CustomException("No suitable model found", sys)

            logging.info(f"Best model found: {best_model_name} with R²: {best_model_score:.4f}")

            best_model.fit(X_train, y_train)

            preprocessor = load_object(self.model_trainer_config.preprocessor_path)

            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = []  # Fallback

            model_bundle = {
                "model": best_model,
                "expected_features": feature_names
            }

            save_object(self.model_trainer_config.trained_model_file_path, model_bundle)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"Model R² Score on Test Set: {r2_square:.4f}")
            return r2_square

        except Exception as e:
            raise CustomException(str(e), sys)
