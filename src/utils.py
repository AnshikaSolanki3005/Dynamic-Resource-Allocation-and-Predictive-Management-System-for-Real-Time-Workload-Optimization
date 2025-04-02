import os
import sys
import pandas as pd
import numpy as np
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from src.exception import CustomException

def save_object(file_path, obj):
    """Save object using dill."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params, cv=5, n_jobs=-1, verbose=2, refit=True):
    """Evaluate multiple models using GridSearchCV and return a performance report."""
    try:
        report = {}

        for model_name, model in models.items():
            param_grid = params.get(model_name, {})

            gs = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs, verbose=verbose, refit=refit)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Load object using dill."""
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)
