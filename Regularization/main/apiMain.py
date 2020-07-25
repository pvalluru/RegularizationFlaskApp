# from main import DataCollecting
from Regularization import Regularization
from flask import Flask

app = Flask(__name__)

@app.route("/api/LinearRegression", methods=['GET'])
def getAccForRegression():
    regularization = Regularization()
    ReadDataFrame = regularization.readDataSet()
    if ReadDataFrame.empty:
        return {"info":"We are not able to read the data from Sklearn"}
    else:
        regularization.saveDataSetToFolder(ReadDataFrame)
        # as we have the data frame we need to divide the
        # data frame into dependent and independent variables
        featuresColumns, targetColum = regularization.featureAndTargetColumn(ReadDataFrame)
        if featuresColumns.empty and targetColum.empty:
            return {"info":"Features and targets columns are empty! check the data set."}
        else:
            # Do Cross Validation for linear regression on the given data set
            meanSquaredError = regularization.doLinerRegression(featuresColumns, targetColum)
            return {"Mean Squared Error": meanSquaredError, "Model": "LinearReqression"}

@app.route("/api/LassoRegression", methods=['GET'])
def getAccForLassoReg():
    regularization = Regularization()
    ReadDataFrame = regularization.readDataSet()
    if ReadDataFrame.empty:
        return {"info": "We are not able to read the data from Sklearn"}
    else:
        # as we have the data frame we need to divide the
        # data frame into dependent and independent variables
        featuresColumns, targetColum = regularization.featureAndTargetColumn(ReadDataFrame)
        if featuresColumns.empty and targetColum.empty:
            return {"info": "Features and targets columns are empty! check the data set."}
        else:
            # Do Cross Validation for linear regression on the given data set
            result = regularization.doLassoLinerRegression(featuresColumns, targetColum)
            return result

@app.route("/api/RidgeRegression", methods=['GET'])
def getAccForRidgeReg():
    regularization = Regularization()
    ReadDataFrame = regularization.readDataSet()
    if ReadDataFrame.empty:
        return {"info": "We are not able to read the data from Sklearn"}
    else:
        # as we have the data frame we need to divide the
        # data frame into dependent and independent variables
        featuresColumns, targetColum = regularization.featureAndTargetColumn(ReadDataFrame)
        if featuresColumns.empty and targetColum.empty:
            return {"info": "Features and targets columns are empty! check the data set."}
        else:
            # Do Cross Validation for linear regression on the given data set
            result = regularization.doRidgeLinerRegression(featuresColumns, targetColum)
            return result

if __name__ == '__main__':
    app.run()