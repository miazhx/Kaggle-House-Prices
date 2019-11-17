import numpy as np
import pandas as pd


from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from math import sqrt
from scipy import stats
from scipy.stats import norm


import matplotlib.pyplot as plt
import matplotlib as matplotlib
import seaborn as sns
%matplotlib inline



def data_process(test=False):
    
    
    # Read file for train or test.
    
    df_raw_train = pd.read_csv('train.csv',index_col=0)
    df_raw_test = pd.read_csv('test.csv',index_col=0)
    
#     print(df_raw_train.shape)
#     print(df_raw_test.shape)
        
    # Remove outliers in training set.   
    outlier_list_scatter = [524, 1299]
    outlier_list_hard_to_fit = [463, 31, 534, 1433, 739, 1159, 108, 1231, 971, 1424 ]
    outlier_list = outlier_list_scatter + outlier_list_hard_to_fit
    df_raw_train = df_raw_train.drop(outlier_list)
    
    # Store the sale price information
    sale_price_train = df_raw_train['SalePrice']
    
    # Merge train and test df together for later process
    df_processed = pd.concat([df_raw_train, df_raw_test], sort=True)
    

    # Combine bathroom quanlitity 
    df_processed['BsmtBath'] = df_processed.BsmtFullBath + df_processed.BsmtHalfBath * 0.5
    df_processed['Bath'] = df_processed.FullBath + df_processed.HalfBath * 0.5
       
    
    ## Drop multicollinear columns 
    df_processed = df_processed.drop(['BsmtFullBath', 'BsmtHalfBath','FullBath','HalfBath'], axis=1)
    
    
    ## Missing values
    
    # 259 LotFrontage  - replace missing value with 0 
#     df_processed.LotFrontage = df_processed.LotFrontage.fillna(0)
    df_processed["LotFrontage"] = df_processed.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

    # 1369 Alley - replace with None
    df_processed.Alley = df_processed.Alley.fillna('None')

    # 8 MasVnrType and MasVnrArea - replace MasVnrType with None and MasVnrArea with 0
    df_processed.MasVnrType = df_processed.MasVnrType.fillna('None')
    df_processed.MasVnrArea = df_processed.MasVnrArea.fillna(0)

    # 37 basement: BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2- replace with None
    df_processed.BsmtQual = df_processed.BsmtQual.fillna('None')
    df_processed.BsmtCond = df_processed.BsmtCond.fillna('None')
    df_processed.BsmtExposure = df_processed.BsmtExposure.fillna('None')
    df_processed.BsmtFinType1 = df_processed.BsmtFinType1.fillna('None')
    df_processed.BsmtFinType2 = df_processed.BsmtFinType2.fillna('None')
    df_processed.TotalBsmtSF = df_processed.TotalBsmtSF.fillna(0)
    

    # 690 FireplaceQu - replace with None
    df_processed.FireplaceQu = df_processed.FireplaceQu.fillna('None')

    # 81 Garage: GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond - replace with None and year with 0 
    df_processed.GarageType = df_processed.GarageType.fillna('None')
    df_processed.GarageFinish = df_processed.GarageFinish.fillna('None')
    df_processed.GarageQual = df_processed.GarageQual.fillna('None')
    df_processed.GarageCond = df_processed.GarageCond.fillna('None')
    df_processed.GarageYrBlt = df_processed.GarageYrBlt.fillna(0)

    # 1453 PoolQC - replace with None
    df_processed.PoolQC = df_processed.PoolQC.fillna('None')

    # 1179 Fence - replace with None
    df_processed.Fence = df_processed.Fence.fillna('None')

    # 1406 MiscFeature - replace with None    
    df_processed.MiscFeature = df_processed.MiscFeature.fillna('None')

    # 1 Electrical
    df_processed = df_processed[pd.notnull(df_processed.Electrical)]
    
    #Missing Value only in test data 
    
    # MSZoning (The general zoning classification) : 'RL' is by far the most common value. So we can fill in missing values with 'RL'
    df_processed['MSZoning'] = df_processed['MSZoning'].fillna(df_processed['MSZoning'].mode()[0])

    # Utilities : For this categorical feature all records are "AllPub", except for one "NoSeWa" and 2 NA . Since the house with 'NoSewa' is in the training set, this feature won't help in predictive modelling. We can then safely remove it.
    df_processed.drop(['Utilities'], axis=1,inplace=True)

    # Exterior1st and Exterior2nd : Again Both Exterior 1 & 2 have only one missing value. We will just substitute in the most common string
    df_processed['Exterior1st'] = df_processed['Exterior1st'].fillna(df_processed['Exterior1st'].mode()[0])
    df_processed['Exterior2nd'] = df_processed['Exterior2nd'].fillna(df_processed['Exterior2nd'].mode()[0]) 
    
    # BsmtFinSF1, BsmtFinSF2, BsmtUnfSF, TotalBsmtSF, BsmtBath : missing values are likely zero for having no basement
    for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtBath'):
        df_processed[col] = df_processed[col].fillna(0)    
    
    #Garage Cars 
    df_processed.GarageCars = df_processed.GarageCars.fillna(0) 
    
    # SaleType : Fill in again with most frequent which is "WD"
    df_processed['SaleType'] = df_processed['SaleType'].fillna(df_processed['SaleType'].mode()[0])
    
    # KitchenQual: Only one NA value, and same as Electrical, we set 'TA' (which is the most frequent) for the missing value in KitchenQual.
    df_processed['KitchenQual'] = df_processed['KitchenQual'].fillna(df_processed['KitchenQual'].mode()[0])    
    
    # Functional : data description says NA means typical
    df_processed["Functional"] = df_processed["Functional"].fillna("Typ")    
    

    # drop GarageArea - higher correlation than GarageACars, results are better as well
    df_processed = df_processed.drop(['GarageArea'], axis=1) 
    df_processed = df_processed.drop(['MiscFeature'], axis=1) 
#     df_processed = df_processed.drop(['1stFlrSF'], axis=1) 
    df_processed = df_processed.drop(['TotRmsAbvGrd'], axis=1) 

    
    # Feature Transformation - take the logarithm of the features.
    #Linear_Num_Cols = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'LotArea', 'GarageArea', 'TotRmsAbvGrd', 'TotalSF', 'BsmtFinSF1']
    df_processed.GrLivArea = np.log(df_processed.GrLivArea)
    df_processed.TotalBsmtSF = np.log(df_processed.TotalBsmtSF+1)
#     df_processed.LotArea = np.log(df_processed.LotArea) -- performance decreases
#     df_processed.GarageArea = np.log(df_processed.GarageArea) -- will drop column 



    # Categorical Features Processsing

    # MSSubClass processing - MSSubClass 20-90 contains only duplicate information with HouseStyle and YearBuilt.
    df_processed['MSSubClass'] = df_processed['MSSubClass'].replace(['20','30','40','45','50','60','70','75','80','85'], '0')

    # Convert numerical to categorical. 
    df_processed['MSSubClass'] = df_processed['MSSubClass'].astype(str)

    #Encode some categorical features as ordered numbers when there is information in the order.
    df_processed = df_processed.replace({"Alley" : {"None":0,"Grvl" : 1, "Pave" : 2},
                       "BsmtCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "BsmtExposure" : {"None" : 0,"No":1, "Mn" : 2, "Av": 3, "Gd" : 4},
                       "BsmtFinType1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtFinType2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4,
                                         "ALQ" : 5, "GLQ" : 6},
                       "BsmtQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                       "ExterCond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "ExterQual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
                       "FireplaceQu" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "Functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5,
                                       "Min2" : 6, "Min1" : 7, "Typ" : 8},
                       "GarageCond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "GarageQual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "HeatingQC" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "KitchenQual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                       "LandSlope" : {"Sev" : 3, "Mod" : 2, "Gtl" : 1},
                       "LotShape" : {"IR3" : 4, "IR2" : 3, "IR1" : 2, "Reg" : 1},
                       "PavedDrive" : {"N" : 0, "P" : 1, "Y" : 2},
                       "PoolQC" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
                       "Street" : {"Grvl" : 1, "Pave" : 2}})
    

    
    # Year processing 
    # Combine year sold with year build to year old
    df_processed['YearsOld']  = df_processed['YrSold'] - df_processed['YearBuilt']
    df_processed = df_processed.drop(['YearBuilt'], axis=1)

    # Combine YrSold and YearRemodAdd
    df_processed['YearSinceRemodel'] = df_processed['YrSold'] - df_processed['YearRemodAdd']
    df_processed = df_processed.drop(['YearRemodAdd'], axis=1)
    df_processed = df_processed.drop(['YrSold'], axis=1)
    
    # Missing rate greater than 47%, and low correlation with sale price
    df_processed = df_processed.drop(['FireplaceQu'], axis=1)

    # PoolQC has .99 missing value. drop will lower rmse
    df_processed = df_processed.drop(['PoolQC'], axis=1)
    
#     MiscVal is 0 when MiscFeature is missing. drop will lower rmse a little
#     df_processed = df_processed.drop(['MiscVal'], axis=1)
    
    #Get Dummies 
    df_processed = pd.get_dummies(df_processed, columns=df_processed.select_dtypes(include=['object']).columns, drop_first=True)

    # Split train and test data sets
    df_processed_train = df_processed[df_processed.index <= 1460].copy()
    df_processed_test = df_processed[df_processed.index > 1460].copy()
    
    # take log on price
    sale_price_train = np.log(sale_price_train)
    df_processed_train['SalePrice'] = sale_price_train    
    
    if test is False:
        return df_processed_train
    if test is True:
        return df_processed_test