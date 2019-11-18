#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd

from math import sqrt
from scipy import stats
from scipy.stats import norm

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

import os
import warnings
import sys

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

from data_processing import data_process

def train_linear( ):

    
    #from data_processing import data_process

    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        return rmse, r2


    warnings.filterwarnings("ignore")
    np.random.seed(40)

#     # Read the file
#     try:
#         df_raw = pd.read_csv('train.csv',index_col=0)
#     except Exception as e:
#         logger.exception(
#             "Unable to download training & test CSV, check your internet connection. Error: %s", e)
        
    # Data processing.
    df_processed = data_process()
    
    # The predicted column is "SalePrice" , split the data into training and test sets. (0.75, 0.25) split.
    x_m = df_processed.drop(["SalePrice"], axis=1)
    y_m = df_processed.loc[:,'SalePrice']
    X_train, X_test, y_train, y_test = train_test_split(x_m, y_m, test_size=0.2, random_state=42)
      
    
    # Execute linear regression
    with mlflow.start_run():
        ols = LinearRegression()
        ols.fit(X_train, y_train)

        # Evaluate Metrics
        rmse = sqrt(mean_squared_error(y_test, ols.predict(X_test)))

        # Print out metrics
        print("R^2 for train set: %f" %ols.score(X_train, y_train))
        print('-'*50)
        print("R^2 for test  set: %f" %ols.score(X_test, y_test))
        print('-'*50)
        print("RMSE for test  set: %f" %rmse)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("model", 'linear_regression')
        mlflow.log_metric("r2_train", ols.score(X_train, y_train))
        mlflow.log_metric("r2_test", ols.score(X_test, y_test))
        mlflow.log_metric("rmse", rmse)



        mlflow.sklearn.log_model(ols, "model")
        
def train_elasticnet(N_alpha, N_rho):


    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2_test = r2_score(actual, pred)
        return rmse, r2_test


    warnings.filterwarnings("ignore")
    np.random.seed(40)

#     # Read the file
#     try:
#         df_raw = pd.read_csv('train.csv',index_col=0)
#     except Exception as e:
#         logger.exception(
#             "Unable to download training & test CSV, check your internet connection. Error: %s", e)
        
    # Data processing.
    df_processed = data_process()
    
    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(df_processed)

    # The predicted column is "SalePrice" .
    train_x = train.drop(["SalePrice"], axis=1)
    test_x = test.drop(["SalePrice"], axis=1)
    train_y = train[["SalePrice"]]
    test_y = test[["SalePrice"]]

    # Set default values if no N_alpha is provided
    if int(N_alpha) is None:
        N_alpha = 50
    else:
        N_alpha = int(N_alpha)

    # Set default values if no N_rho is provided
    if int(N_rho) is None:
        N_rho = 11
    else:
        N_rho = int(N_rho)
    
    alphaRange = np.logspace(-3, -1, N_alpha)
    rhoRange   = np.linspace(0,0.4, N_rho) # we avoid very small rho by starting at 0.1
    scores     = np.zeros((N_rho, N_alpha))
      
    
    # Execute ElasticNet
    for alphaIdx, alpha in enumerate(alphaRange):
        for rhoIdx, rho in enumerate(rhoRange):
            with mlflow.start_run():
                lr = ElasticNet(alpha=alpha, l1_ratio=rho, normalize=False)
                lr.fit(train_x, train_y)
                r2_train = lr.score(train_x, train_y)
        
        # Training Model Performances Evaluate Metrics
#         predicted_qualities = lr.predict(test_x)
#         (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)        

        # Evaluate Metrics
                predicted_qualities = lr.predict(test_x)
                (rmse, r2_test) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
                print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, rho))
                print("  RMSE: %s" % rmse)
                print("  Train R2: %s" % r2_train)
                print("  Test R2: %s" % r2_test)

        # Log parameter, metrics, and model to MLflow
                mlflow.log_param("model", 'elasticnet')
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", rho)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2_test", r2_test)
                mlflow.log_metric("r2_train", r2_train)
                #mlflow.log_metric("mae", mae)

                mlflow.sklearn.log_model(lr, "model")

