from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation, DataTransformationConfig
import sys

if __name__=="__main__":
    logging.info("The execution has started.")

    try:
        data_ingestion=DataIngestion()
        # Data_Ingestion_Config=DataIngestionConfig()
        train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()

        # data_tranformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)



    except Exception as e:
        logging.info("Custom exception")
        raise CustomException(e, sys)