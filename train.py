import sqlite3
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import common

def load_train_data(path):
    print(f"Reading train data from the database: {path}")
    con = sqlite3.connect(path)
    data_train = pd.read_sql('SELECT * FROM train', con)
    con.close()
    data_train['trip_duration'] =np.log1p(data_train['trip_duration'])
    X = data_train.drop(columns=['trip_duration'])
    y = data_train['trip_duration']
    return X, y

def fit_model(X_train, y_train,prepocess_resuls):
    print(f"Fitting a model")


    model = prepocess_resuls[0].fit(X_train[prepocess_resuls[1]], y_train)
    y_pred_train = model.predict(X_train[prepocess_resuls[1]])
    print("Train RMSE = %.4f" % mean_squared_error(y_train, y_pred_train, squared=False))

    return model

if __name__ == "__main__":

    X_train, y_train = load_train_data(common.DB_PATH)
    prepocess_resuls = common.preprocess_data()
    model = fit_model(X_train, y_train,prepocess_resuls)
    common.persist_model(model, common.MODEL_PATH)
