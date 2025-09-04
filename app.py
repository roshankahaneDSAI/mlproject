from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion, DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformation, DataTransformationConfig
from src.mlproject.components.model_trainer import ModelTrainer, ModelTrainerConfig
from src.mlproject.pipelines.prediction_pipeline import CustomData, PredictPipeline

import sys

## Flask App Routing

from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

## Create a simple  flask application
app=Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/prediction", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        try:
            logging.info("The Inference pipeline has triggered.")
            data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

            )
            pred_df=data.get_data_as_data_frame()
            print(pred_df)
            logging.info("The Inference dataframe has created.")
            print("Before Prediction")

            predict_pipeline=PredictPipeline()
            print("Mid Prediction")
            results=predict_pipeline.predict(pred_df)
            print("after Prediction")
            logging.info("The Inference pipeline prediction is successfull.")

            return render_template('home.html',results=results[0])

        except Exception as e:
            logging.info("Custom exception")
            raise CustomException(e, sys)


if __name__=="__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=True)
    except Exception as e:
        logging.info("Custom exception")
        raise CustomException(e, sys)

    # logging.info("The execution has started.")

    # try:
    #     data_ingestion=DataIngestion()
    #     # Data_Ingestion_Config=DataIngestionConfig()
    #     train_data_path, test_data_path=data_ingestion.initiate_data_ingestion()

    #     # data_tranformation_config=DataTransformationConfig()
    #     data_transformation=DataTransformation()
    #     train_arr, test_arr,_=data_transformation.initiate_data_transformation(train_data_path, test_data_path)

    #     model_trainer=ModelTrainer()
    #     r2_square = model_trainer.initiate_model_trainer(train_arr, test_arr)
    #     print("The r2_square is {}".format(r2_square))

    # except Exception as e:
    #     logging.info("Custom exception")
    #     raise CustomException(e, sys)