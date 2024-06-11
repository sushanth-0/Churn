import sys
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from Customer_Churn.exception import CustomerChurnException
from Customer_Churn.logger import logging
from Customer_Churn.utils.main_utils import load_numpy_array_data, read_yaml_file, load_object, save_object
from Customer_Churn.entity.config_entity import ModelTrainerConfig
from Customer_Churn.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from Customer_Churn.entity.estimator import CustomerChurnmodel

class ModelTrainer:
    def __init__(self, data_transformation_artifact, model_trainer_config):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_best_model(self, X_train, y_train):
        models = {
            "KNeighborsClassifier": (KNeighborsClassifier(), {
                "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                "weights": ['uniform', 'distance'],
                "n_neighbors": [3, 4, 5, 7, 9]
            }),
            "RandomForestClassifier": (RandomForestClassifier(), {
                "max_depth": [10, 12, None, 15, 20],
                "max_features": ['sqrt', 'log2', None],
                "n_estimators": [10, 50, 100, 200]
            }),
            "XGBClassifier": (XGBClassifier(objective='binary:logistic'), {
                "max_depth": range(3, 10, 2),
                "min_child_weight": range(1, 6, 2)
            }),
            "CatBoostClassifier": (CatBoostClassifier(verbose=0), {
                "iterations": [100, 200, 300],
                "learning_rate": [0.01, 0.1, 0.2],
                "depth": [4, 6, 8]
            })
        }

        best_model = None
        best_score = 0
        best_params = None
        best_model_name = None

        for model_name, (model, params) in models.items():
            logging.info(f"**************************************************")
            logging.info(f"Training {model_name}")
            logging.info(f"**************************************************")
            grid_search = GridSearchCV(model, params, cv=3, verbose=3, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            logging.info(f"Best {model_name} params: {grid_search.best_params_}")
            logging.info(f"Best {model_name} score: {grid_search.best_score_}")
            if grid_search.best_score_ > best_score:
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_
                best_params = grid_search.best_params_
                best_model_name = model_name

        logging.info(f"Best model: {best_model_name}")
        logging.info(f"Best parameters: {best_params}")
        logging.info(f"Best score: {best_score}")

        return best_model, best_score, best_params, best_model_name

    def get_model_object_and_report(self, train: np.array, test: np.array):
        try:
            x_train, y_train, x_test, y_test = train[:, :-1], train[:, -1], test[:, :-1], test[:, -1]
            best_model, best_score, best_params, best_model_name = self.get_best_model(X_train=x_train, y_train=y_train)
            y_pred = best_model.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            metric_artifact = ClassificationMetricArtifact(f1_score=f1, precision_score=precision, recall_score=recall)
            logging.info(f"Best model selected: {best_model_name}")
            logging.info(f"Model accuracy: {accuracy}")
            logging.info(f"Model F1 score: {f1}")
            logging.info(f"Model precision: {precision}")
            logging.info(f"Model recall: {recall}")
            return best_model, best_score, metric_artifact, best_model_name
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")
        try:
            train_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_file_path)
            best_model, best_score, metric_artifact, best_model_name = self.get_model_object_and_report(train=train_arr, test=test_arr)
            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            if best_score < self.model_trainer_config.expected_accuracy:
                logging.info("No best model found with score more than base score")
                raise Exception("No best model found with score more than base score")

            customer_churn_model = CustomerChurnmodel(preprocessing_object=preprocessing_obj,
                                                      trained_model_object=best_model)
            logging.info(f"Created CustomerChurnModel object with preprocessor and model: {best_model_name}")
            save_object(self.model_trainer_config.trained_model_file_path, customer_churn_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                metric_artifact=metric_artifact,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            logging.info(f"Best model stored in model.pkl file: {best_model_name}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomerChurnException(e, sys) from e