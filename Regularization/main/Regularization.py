# Import libraries
import os
import sys

env_p = sys.prefix  # path to the env
print("Env. path: {}".format(env_p))

new_p = ''
for extra_p in (r"Library\mingw-w64\bin",
    r"Library\usr\bin",
    r"Library\bin",
    r"Scripts",
    r"bin"):
    new_p +=  os.path.join(env_p, extra_p) + ';'

os.environ["PATH"] = new_p + os.environ["PATH"]  # set it for Python
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import pandas as pd

class Regularization():
    def __init__(self):
        self.dataset = pd.DataFrame()

    # this function will read the data from Sklearn
    # return as a dataframe
    def readDataSet(self):
        bostenDataSet = pd.DataFrame()
        try:
            loadDataSet = load_boston()
            bostenDataSet = pd.DataFrame(loadDataSet.data, columns=[loadDataSet.feature_names])
            bostenDataSet['Price'] = loadDataSet.target
        except Exception as exc:
            print("Exception caught while reading the data from sklearn: "+ str(exc))
        return bostenDataSet

    def saveDataSetToFolder(self, dataSet):
        try:
            if not os.path.exists("../Data/"):
                os.mkdir("../Data/")
                print("Create a DATA folder")
            dataSet.to_csv("../Data/bostonDataSet.csv")
        except Exception as exc:
            print("Exception caught while writing the data set to CSV:"+ str(exc))

    def featureAndTargetColumn(self, dataSet=None):
        featureColumn = None
        targetColumn = None
        try:
            if dataSet is None or dataSet.empty:
                print("Data set is empty or Data set is not passed")
            else:
                featureColumn, targetColumn = dataSet.iloc[:,:-1], dataSet.iloc[:,-1]
        except Exception as exc:
            print("Exception caught while taking features and target columns: "+ str(exc))
        return featureColumn, targetColumn

    def doLinerRegression(self, X_label, Y_label):
        meanSquaredError = 0
        try:
            LinearReg = LinearRegression()
            mse = cross_val_score(LinearReg, X_label, Y_label, cv=5, scoring='neg_mean_squared_error')
            meanSquaredError = np.mean(mse)
        except Exception as exc:
            print("Exception caught while performing Cross validation for Liner Regression."+ str(exc))
        return meanSquaredError

    def doLassoLinerRegression(self, X_label, Y_label):
        meanSquaredError = 0
        try:
            LassoReg = Lasso()
            parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 40, 45, 50, 55, 100]}
            lassoRegression = GridSearchCV(LassoReg, parameters, cv=5, scoring='neg_mean_squared_error')
            lassoRegression.fit(X_label, Y_label)
        except Exception as exc:
            print("Exception caught while performing Cross validation for Liner Regression."+ str(exc))
        return {"Best Score": lassoRegression.best_score_, "Best Parameter": lassoRegression.best_params_}

    def doRidgeLinerRegression(self, X_label, Y_label):
        meanSquaredError = 0
        try:
            ridgeReg = Ridge()
            parameters = {'alpha': [1e-15,1e-10,1e-8,1e-3,1e-2, 1,5,10,20,30,40,45,50,55,100]}
            ridgeRegression = GridSearchCV(ridgeReg, parameters, cv=5, scoring='neg_mean_squared_error')
            ridgeRegression.fit(X_label, Y_label)
        except Exception as exc:
            print("Exception caught while performing Cross validation for Liner Regression."+ str(exc))
        return {"Best Score": ridgeRegression.best_score_, "Best Parameter": ridgeRegression.best_params_}
