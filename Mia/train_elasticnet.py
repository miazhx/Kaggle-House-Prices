#!/usr/bin/env python
# coding: utf-8

# In[1]:


# MLflow Training
def train(N_alpha, N_rho):
    import os
    import warnings
    import sys

    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNet

    import mlflow
    import mlflow.sklearn
    
    import logging
    logging.basicConfig(level=logging.WARN)
    logger = logging.getLogger(__name__)
    
    from data_processing import data_process

    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the file
    try:
        df_raw = pd.read_csv('train.csv',index_col=0)
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e)
        
    # Data processing.
    df_processed = data_process(df_raw)
    
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
    
    alphaRange = np.logspace(-3, -2, N_alpha)
    rhoRange   = np.linspace(0,1, N_rho) # we avoid very small rho by starting at 0.1
    scores     = np.zeros((N_rho, N_alpha))
      
    
    # Execute ElasticNet
    for alphaIdx, alpha in enumerate(alphaRange):
        for rhoIdx, rho in enumerate(rhoRange):
            with mlflow.start_run():
                lr = ElasticNet(alpha=alpha, l1_ratio=rho, normalize=False)
                lr.fit(train_x, train_y)
                scores[rhoIdx, alphaIdx] = lr.score(train_x, train_y)
        
        # Training Model Performances Evaluate Metrics
#         predicted_qualities = lr.predict(test_x)
#         (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)        

        # Evaluate Metrics
                predicted_qualities = lr.predict(test_x)
                (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
                print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, rho))
                print("  RMSE: %s" % rmse)
                #print("  MAE: %s" % mae)
                print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("l1_ratio", rho)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                #mlflow.log_metric("mae", mae)

                mlflow.sklearn.log_model(lr, "model")


# In[2]:


train(50,11)

