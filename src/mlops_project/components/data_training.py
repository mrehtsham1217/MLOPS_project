import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.mlops_project.logger import  logging
from src.mlops_project.exception import CustomException
from src.mlops_project.utils import save_model,evaluate_models

@dataclass
class DataTrainingConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")
class DataTraining:

    def __init__(self):
        self.data_training = DataTrainingConfig()

    def eval_metric(self,actual,prediction):
        rmse = np.sqrt(mean_squared_error(actual,prediction))
        mae = mean_absolute_error(actual,prediction)
        r2 = r2_score(actual,prediction)
        return (
            rmse,
            mae,
            r2,
        )

    def data_training_initiate(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            parmas = {
                "Decision Tree": {
                    'criterion': ["squared_error", "friedman_mse", "absolute_error"],
                    'splitter': ["best", "random"],
                    'max_features': ['sqrt', 'log2']
                },
                "Random Forest": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'max_features': ['sqrt', 'log2', None],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    'learning_rate': [.1, .01, .05, .001],
                    'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8, 10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [.1, .01, 0.5, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                }
            }

            model_report:dict = evaluate_models(X_train, y_train, X_test, y_test,models,parmas)
            best_model_score = max(sorted(model_report.values()))

            best_models_names = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_models_names]
            print("This is the best model:")
            print(best_models_names)

            model_names = list(parmas.keys())

            actual_model = ""

            for model in model_names:
                if best_models_names == model:
                    actual_model = actual_model + model

            best_params = parmas[actual_model]
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_model(
               file_path=self.data_training.trained_model_file_path,
                model=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            logging.info("Custom error")
            raise CustomException(sys, e)




