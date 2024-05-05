from src.mlops_project.logger import logging
from src.mlops_project.exception import CustomException
from src.mlops_project.components.data_ingestion import DataIngestionConfig,DataIngestion

#import data transformation
from src.mlops_project.components.data_tranformation import DataTransformationConfig,DataTransformation
import sys
from src.mlops_project.components.data_training import DataTrainingConfig,DataTraining
if __name__ == "__main__":
    logging.info("Executing has started")
    try:
        data_ingestion = DataIngestion()
        train_datapath,test_datapath =  data_ingestion.initiate_data_ingestion()
        # print(train_datapath.shape)
        # print(test_datapath.shape)

        data_tranformation = DataTransformation()
        train_array,test_array = data_tranformation.initiate_data_transformation(train_datapath,test_datapath)
        data_training = DataTraining()
        print(data_training.data_training_initiate(train_array,test_array))
    except Exception as e:
        logging.info(f"Customer Exception")
        raise CustomException (e,sys)