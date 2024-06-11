from Customer_Churn.entity.config_entity import ModelEvaluationConfig
from Customer_Churn.entity.artifact_entity import ModelTrainerArtifact, DataIngestionArtifact, ModelEvaluationArtifact
from sklearn.metrics import f1_score
from Customer_Churn.exception import CustomerChurnException
from Customer_Churn.constants import TARGET_COLUMN
from Customer_Churn.logger import logging
import sys
import pandas as pd
from typing import Optional
from Customer_Churn.entity.s3_estimator import CustomerChurnEstimator
from dataclasses import dataclass
from Customer_Churn.entity.estimator import CustomerChurnmodel
from Customer_Churn.entity.estimator import TargetValueMapping

def convert_total_charges(df, column_name='TotalCharges', fill_value=0, dtype=float):
    """
    Converts a specified column to a numeric type, coerces errors to NaN, 
    fills NaN values, and converts the column to a specified data type.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the column to convert.
    column_name (str): The name of the column to convert. Default is 'TotalCharges'.
    fill_value (int or float): The value to use for filling NaNs. Default is 0.
    dtype (type): The data type to convert the column to. Default is float.

    Returns:
    pd.DataFrame: The DataFrame with the converted column.
    """
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df[column_name].fillna(fill_value, inplace=True)
    df[column_name] = df[column_name].astype(dtype)
    return df


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig, data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def get_best_model(self) -> Optional[CustomerChurnEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model in production
        
        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            CustomerChurn_estimator = CustomerChurnEstimator(bucket_name=bucket_name,
                                                             model_path=model_path)

            if CustomerChurn_estimator.is_model_present(model_path=model_path):
                return CustomerChurn_estimator
            return None
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model 
                        with production model and choose best model 
        
        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Convert the TotalCharges column
            test_df = convert_total_charges(test_df, 'TotalCharges')

            x, y = test_df.drop(TARGET_COLUMN, axis=1), test_df[TARGET_COLUMN]
            y = y.replace(TargetValueMapping()._asdict())

            trained_model_f1_score = self.model_trainer_artifact.metric_artifact.f1_score

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                y_hat_best_model = best_model.predict(x)
                best_model_f1_score = f1_score(y, y_hat_best_model)
            
            tmp_best_model_score = 0 if best_model_f1_score is None else best_model_f1_score
            result = EvaluateModelResponse(trained_model_f1_score=trained_model_f1_score,
                                           best_model_f1_score=best_model_f1_score,
                                           is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                                           difference=trained_model_f1_score - tmp_best_model_score
                                           )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise CustomerChurnException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation
        
        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """  
        try:
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference)

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
