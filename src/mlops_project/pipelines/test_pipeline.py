import os
import sys
import pandas as pd
from src.mlops_project.utils import load_object
from src.mlops_project.logger import logging
from src.mlops_project.exception import CustomException
from sklearn.preprocessing import OneHotEncoder


class PredictionPipelines:
    def __init__(self):
        pass

    def prediction(self, features):
        try:
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            logging.info("Custom error")
            raise CustomException(sys, e)


class CustomerData:
    def __init__(self,
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 writing_score: int,
                 reading_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score

    def get_data_frame(self):
        try:
            customer_data_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "writing_score": [self.writing_score],
                "reading_score": [self.reading_score]
            }
            return pd.DataFrame(customer_data_dict)
        except Exception as e:
            logging.info("Custom error")
            raise CustomException(sys, e)
