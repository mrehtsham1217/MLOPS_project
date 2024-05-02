import os
import pymysql
from dotenv import load_dotenv
from src.mlops_project.logger import logging
from src.mlops_project.exception import CustomException
import pandas as pd
import sys
import pickle

load_dotenv()
host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

def read_sql():
    logging.info("Reading data from database")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info("Connection is established {}".format(mydb))
        df = pd.read_sql_query('SELECT * FROM raw', mydb)
        print(df.head())
        return df
    except Exception as e:
        # logging.info(e)
        raise CustomException(sys,e)
def save_model(file_path,model):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logging.info("saving the model using pickle.dump")
    except Exception as e:
        logging.info("Customer Exception")
        raise CustomException(sys,e)


# read_sql()
