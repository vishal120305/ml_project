import os
import sys
import pandas as pd 
import numpy as np
from src.logger import logging, log_error
from src.exception import CustomException   
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from src.utils import save_object, evaluate_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')
    expected_score: float = 0.6
    overfitting_threshold: float = 0.05 
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": XGBRegressor(),
                "CatBoosting": CatBoostRegressor(verbose=False),
                "Linear Regression": LinearRegression(),
                "K-Neighbours": KNeighborsRegressor()
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

           
            best_model_score = max(sorted(model_report.values()))

           
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score < self.model_trainer_config.expected_score:
                log_error("No best model found")
                raise CustomException("No best model found", sys)

            logging.info(f"Best found model on both training and testing dataset is {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        except Exception as e:
            log_error(e)
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    pass
    # --- IGNORE ---
    # train_array = np.load('artifacts/train_transformed.npy')
    # test_array = np.load('artifacts/test_transformed.npy')
    # model_trainer = ModelTrainer()
    # r2_square = model_trainer.initiate_model_trainer(train_array, test
    #                                            )_array)
    # print(r2_square)  
    # --- IGNORE ---
    # --- IGNORE ---
    # train_array = np.load('artifacts/train_transformed.npy')
    # test_array = np.load('artifacts/test_transformed.npy')
    # model_trainer = ModelTrainer()
    # r2_square = model_trainer.initiate_model_trainer(train_array, test
    #                                            )_array)
    # print(r2_square)