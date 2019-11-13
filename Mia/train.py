#!/usr/bin/env python
# coding: utf-8

# # MLflow Training 
# 
# This `train.pynb` Jupyter notebook predicts the housing price using [sklearn.linear_model.ElasticNet](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html).  
# 
# > This is the Jupyter notebook version of the `train.py` example
# 
# Attribution
# * The MLflow code used in this module is from https://www.mlflow.org/docs/latest/tutorial.html
# 

# In[44]:


# MLflow Training
def train(in_alpha, in_l1_ratio):
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

    # Set default values if no alpha is provided
    if float(in_alpha) is None:
        alpha = 0.5
    else:
        alpha = float(in_alpha)

    # Set default values if no l1_ratio is provided
    if float(in_l1_ratio) is None:
        l1_ratio = 0.5
    else:
        l1_ratio = float(in_l1_ratio)

    # Useful for multiple runs (only doing one run in this sample notebook)    
    with mlflow.start_run():
        # Execute ElasticNet
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Evaluate Metrics
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        # Print out metrics
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # Log parameter, metrics, and model to MLflow
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        mlflow.sklearn.log_model(lr, "model")


# In[48]:


train(0.5, 0.5)


# In[49]:


train(0.2, 0.2)


# In[51]:


train(0.1, 0.2)


# In[ ]:


# alphas = np.arange(0,20)
# ridge.set_params(normalize=True)
# coefs  = []
# scores = []
# for alpha in alphas:
#         ridge.set_params(alpha=alpha)
#         ridge.fit(house_features, prices)  
#         coefs.append(ridge.coef_)
#         scores.append(ridge.score(house_features, prices))
# coefs = pd.DataFrame(coefs, index = alphas, columns = house_features.columns)  
# coefs.head()

