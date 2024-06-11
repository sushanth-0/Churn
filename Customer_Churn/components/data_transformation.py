import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

from Customer_Churn.constants import TARGET_COLUMN, SCHEMA_FILE_PATH
from Customer_Churn.entity.config_entity import DataTransformationConfig
from Customer_Churn.entity.artifact_entity import DataTransformationArtifact, DataIngestionArtifact, DataValidationArtifact
from Customer_Churn.exception import CustomerChurnException
from Customer_Churn.logger import logging
from Customer_Churn.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file, drop_columns
from Customer_Churn.entity.estimator import TargetValueMapping

def convert_total_charges(df, column_name='TotalCharges', fill_value=0, dtype=float):
    df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
    df[column_name].fillna(fill_value, inplace=True)
    df[column_name] = df[column_name].astype(dtype)
    return df

class DataTransformation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomerChurnException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data read from {file_path} with shape: {df.shape}")
            return df
        except Exception as e:
            raise CustomerChurnException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        try:
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            ordinal_encoder = OrdinalEncoder()
            transform_pipe = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, self._schema_config['oh_columns']),
                    ("Ordinal_Encoder", ordinal_encoder, self._schema_config['or_columns']),
                    ("Transformer", transform_pipe, self._schema_config['transform_columns']),
                    ("StandardScaler", numeric_transformer, self._schema_config['num_features'])
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomerChurnException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if self.data_validation_artifact.validation_status:
                preprocessor = self.get_data_transformer_object()
                train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
                test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)

                # Convert 'TotalCharges' column
                train_df = convert_total_charges(train_df)
                test_df = convert_total_charges(test_df)

                logging.info(f"Train data after conversion has shape: {train_df.shape}")
                logging.info(f"Test data after conversion has shape: {test_df.shape}")

                input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_train_df = train_df[TARGET_COLUMN].replace(TargetValueMapping()._asdict())
                input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
                target_feature_test_df = test_df[TARGET_COLUMN].replace(TargetValueMapping()._asdict())

                logging.info(f"Applying preprocessing object on training data")
                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                logging.info(f"Train features array shape: {input_feature_train_arr.shape}")

                logging.info(f"Applying preprocessing object on testing data")
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)
                logging.info(f"Test features array shape: {input_feature_test_arr.shape}")

                logging.info("Applying SMOTEENN on Training dataset")
                smt = SMOTEENN(sampling_strategy="minority")
                input_feature_train_final, target_feature_train_final = smt.fit_resample(input_feature_train_arr, target_feature_train_df)
                logging.info(f"Train final features shape after SMOTEENN: {input_feature_train_final.shape}")

                logging.info("Applying SMOTEENN on Testing dataset")
                input_feature_test_final, target_feature_test_final = smt.fit_resample(input_feature_test_arr, target_feature_test_df)
                logging.info(f"Test final features shape after SMOTEENN: {input_feature_test_final.shape}")

                train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
                test_arr = np.c_[input_feature_test_final, np.array(target_feature_test_final)]

                save_object(self.data_transformation_config.transformed_object_file_path, preprocessor)
                save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
                save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
                )
                return data_transformation_artifact
            else:
                raise Exception(self.data_validation_artifact.message)
        except Exception as e:
            raise CustomerChurnException(e, sys) from e
