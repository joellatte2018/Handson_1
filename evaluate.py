import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

import common

def load_test_data(path):
    print(f"Reading test data from the database: {path}")
    con = sqlite3.connect(path)
    data_test = pd.read_sql('SELECT * FROM test', con)
    con.close()
    data_test['trip_duration'] =np.log1p(data_test['trip_duration'])

    X = data_test.drop(columns=['trip_duration'])
    y = data_test['trip_duration']
    return X, y

def evaluate_model(model, X, y,test_features):
    print(f"Evaluating the model")
    y_pred_test = model.predict(X[test_features])
    print("Test RMSE = %.4f" % mean_squared_error(y, y_pred_test, squared=False))
    score = mean_squared_error(y, y_pred_test,squared=False)

    return score

if __name__ == "__main__":

    X_test, y_test = load_test_data(common.DB_PATH)
    model = common.load_model(common.MODEL_PATH)
    prepocess_resuls = common.preprocess_data()
    score_test = evaluate_model(model, X_test, y_test,prepocess_resuls[1])
    print(f"Score on test data {score_test:.2f}")
