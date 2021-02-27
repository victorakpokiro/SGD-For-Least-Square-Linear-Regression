import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from models.linear_regression import LinearRegression
from optimizers.sgd import SGD
from utils.download import *
from utils.file_types import File_type
from utils.constants import data_url
import os
from utils.misc import Normalization, Metrics



def import_power_plant_data():
    if not os.path.isfile(data_url.POWER_PLANT_DATASET_LOCAL_PATH):
        DownloardAndExtractFile().download_and_extract_file(data_url.POWER_PLANT_DATASET_URL, File_type.ZIP)
    #xls = pd.ExcelFile(data_url.POWER_PLANT_DATASET_LOCAL_PATH)
    df = pd.read_excel(
     data_url.POWER_PLANT_DATASET_LOCAL_PATH,
     engine='openpyxl')
    #Do data normalization
    df = Normalization.min_std_norm(df)

    target = df.PE
    df.drop(["PE"], axis=1, inplace=True)

    return df, target


def init_data():
    X, y = import_power_plant_data()
    X, y = X.to_numpy(), y.to_numpy()
    #print(X,y)
    #exit()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle=True, random_state=1234)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    opt = SGD(lr=0.01)
    regressor = LinearRegression(opt, epoch=10000)
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    #print(len(predicted))
    #exit()
    mse_value = Metrics.mse(y_test, predicted)
    print(mse_value)
    #y_pred_line = regressor.predict(X)
    #cmap = plt.get_cmap('viridis')
    #fig = plt.figure(figsize=(8,6))
    #m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    #m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    #plt.plot(X, y_pred_line, color = "black", linewidth=2, label="Predicted")
    plt.show()

def main():
    init_data()

if __name__ == '__main__':
    main()
