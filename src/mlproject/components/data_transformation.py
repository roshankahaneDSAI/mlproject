import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer 
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object
from sklearn.model_selection import train_test_split


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    def get_data_transformer_object(self, num_features, cat_features):
        '''This function is responsible for data transformation'''
        try:
            # Create Column Transformer with 3 types of transformers
            self.num_features = num_features
            self.cat_features = cat_features

            num_pipeline=Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder()),
                ("Scaler", StandardScaler(with_mean=False))
            ])

            logging.info(f"Categorical Columns:{self.cat_features}")
            logging.info(f"Numerical Columns:{self.num_features}")

            preprocessor=ColumnTransformer([
                ("num_pipeline", num_pipeline, self.num_features),
                ("cat_pipeline", cat_pipeline, self.cat_features)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            target_column='math_score'
            # Create Column Transformer with 3 types of transformers
            # num_features = train_df.select_dtypes(exclude="object").columns
            # cat_features = train_df.select_dtypes(include="object").columns

            # define numerical & categorical columns
            num_features = [feature for feature in train_df.columns if train_df[feature].dtype != 'O']
            num_features.remove(target_column)
            cat_features = [feature for feature in train_df.columns if train_df[feature].dtype == 'O']

            # num_features=['reading_score', 'writing_score']
            # cat_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            logging.info("Reading the train and test files")
            preprocessing_obj=self.get_data_transformer_object(num_features, cat_features)

            # Divide the train dataset to independent and dependent feature
            input_features_train_df=train_df.drop(columns=[target_column], axis=1)
            target_features_train_df=train_df[target_column]

            # Divide the test dataset to independent and dependent feature
            input_features_test_df=test_df.drop(columns=[target_column], axis=1)
            target_features_test_df=test_df[target_column]

            logging.info("Applying Preprocessing on training and test dataframe")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_features_test_df)]

            logging.info(f"Saved the preprocessing object")
            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
