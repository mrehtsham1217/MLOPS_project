import sys
import os
import pandas as pd
import numpy as np
from src.mlops_project.logger import logging
from src.mlops_project.exception import CustomException
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.mlops_project.utils import save_model
from dataclasses import dataclass
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    preprocessing_filepath: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation = DataTransformationConfig()

    def get_transformed_object(self):
        try:
            numerical_cols = ['writing_score', 'reading_score']  # Removed the comma
            categorical_cols = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            numerical_pipelines = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scalar", StandardScaler())
            ])

            categorical_pipelines = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", OneHotEncoder(handle_unknown='ignore')),
                ("scalar", StandardScaler(with_mean=False))
            ])

            logging.info(f"{numerical_pipelines} is created for numerical cols")
            logging.info(f"{categorical_pipelines} is created for categorical cols")

            preprocessor = ColumnTransformer([
                ("numerical", numerical_pipelines, numerical_cols),
                ("categorical", categorical_pipelines, categorical_cols)
            ])  # Removed the square brackets

            return preprocessor
        except Exception as e:
            logging.info(f"Customer Exception {e}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading train and test dataset")
            preprocessor_obj = self.get_transformed_object()
            target_col = "math_score"

            # Separate the input and target features of train dataset
            input_features_train_cols = train_df.drop(target_col, axis=1)
            target_features_train_cols = train_df[target_col]

            # Separate the input and target features of test dataset
            input_features_test_cols = test_df.drop(target_col, axis=1)
            target_features_test_cols = test_df[target_col]

            # Apply data transformations on data
            input_features_train_array = preprocessor_obj.fit_transform(input_features_train_cols)
            input_features_test_array = preprocessor_obj.transform(input_features_test_cols)

            # Combine input and target features in array form
            train_array = np.c_[input_features_train_array, np.array(target_features_train_cols)]
            test_array = np.c_[input_features_test_array, np.array(target_features_test_cols)]

            save_model(
                file_path=self.data_transformation.preprocessing_filepath,
                model=preprocessor_obj
            )

            return train_array, test_array
        except Exception as e:
            logging.info("Customer Exception")
            raise CustomException(e, sys)
