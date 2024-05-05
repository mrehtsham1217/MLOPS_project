import os
import sys
from src.mlops_project.logger import logging
from src.mlops_project.exception import CustomException
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.mlops_project.utils import read_sql

@dataclass
class DataIngestionConfig:
    train_datapath = os.path.join("artifacts","train.csv")
    test_datapath = os.path.join("artifacts","test.csv")
    raw_datapath = os.path.join("artifacts","raw.csv")
class DataIngestion:
    def __init__(self):
        self.data_ingestion = DataIngestionConfig()
    def initiate_data_ingestion(self):
        try:
            # df = read_sql()
            df = pd.read_csv("artifacts/raw.csv")
            logging.info("Reading data from database")
            os.makedirs(os.path.dirname(self.data_ingestion.raw_datapath),exist_ok=True)
            # df=pd.read_csv("Notebook/data","raw.csv")
            df.to_csv(self.data_ingestion.raw_datapath,index=False,header=True)
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.data_ingestion.train_datapath,index=False,header=True)
            test_set.to_csv(self.data_ingestion.test_datapath,index=False,header=True)
            return (
                self.data_ingestion.train_datapath,
                self.data_ingestion.test_datapath,
            )

        except Exception as e:
            logging.info(e)
            raise CustomException(sys,e)