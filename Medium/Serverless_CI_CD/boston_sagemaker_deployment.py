import os

import numpy as np
import pandas as pd

from sklearn.datasets import load_boston
import sklearn.model_selection

import boto3

import sagemaker
from sagemaker.amazon.amazon_estimator import image_uris
from sagemaker.predictor import csv_serializer

# Session and roles
session = sagemaker.Session()
role = 'sagemaker-role'
region = 'us-east-1'

# Load Dataset
boston = load_boston()

# Boston Train Test Splits (ignoring validation sets for now)
X_bos_pd = pd.DataFrame(boston.data, columns=boston.feature_names)
Y_bos_pd = pd.DataFrame(boston.target)
X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_bos_pd, Y_bos_pd, test_size=0.33)


prefix = 'boston-xgboost-deploy-hl'

# SageMaker XGBoost Model
container = sagemaker.image_uris.retrieve("xgboost", region, "1.2-1")
xgb = sagemaker.estimator.Estimator(container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance ot use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

xgb.set_hyperparameters(objective='reg:linear', num_round=1)

# Upload Training data to S3
pd.concat([Y_train, X_train], axis=1).to_csv('train.csv')
train_location = session.upload_data('train.csv', key_prefix=prefix)
s3_input_train = sagemaker.session.s3_input(s3_data=train_location, content_type='csv')

# XGB Fit
xgb.fit({'train': s3_input_train})
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

# Get endpoint of xgb_predictor
print(xgb_predictor.endpoint)

# DELETE ENDPOINT IF YOU'RE DONE
# xgb_predictor.delete_endpoint()