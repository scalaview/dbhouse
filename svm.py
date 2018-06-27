import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

def prepare_train_data(syml):
    import models
    hist = pd.read_sql_query('SELECT `date`, close_price, volumefrom AS volume, trend_tag FROM daily_prices WHERE fsymbol="'+syml+'" AND tsymbol="USDT" AND date between "2016-01-01" AND "2017-10-26"', models.engine)
    hist = hist.set_index('date')
    hist.index = pd.to_datetime(hist.index, unit='s')
    diff_1_close_price = hist["close_price"].diff(1).shift(-1)
    hist['diff_1_close_price'] = (diff_1_close_price/hist["close_price"] * 100)

    diff_1_volume = hist["volume"].diff(1).shift(-1)
    hist['diff_1_volume'] = (diff_1_volume/hist["volume"] * 100)
    return hist.fillna(0)

def build_modle():
    C = 3  # SVM regularization parameter
    return {
        # "rbf":  svm.SVC(kernel='rbf', gamma=0.7, C=C),
        # "linear": svm.SVC(kernel='linear', C=C),
        "poly": Pipeline((
            ("scaler", StandardScaler()),
            ("svm_clf", svm.SVC(kernel="poly", degree=3, coef0=1, C=5))
            )),
        "rbf": Pipeline((
            ("scaler", StandardScaler()),
            ("svm_clf", svm.SVC(kernel="rbf", gamma=0.7, C=C))
            )),
        "linear": Pipeline((
            ("scaler", StandardScaler()),
            ("svm_clf", svm.SVC(kernel="linear", C=C))
            )),
        # "poly": svm.SVC(kernel='poly', degree=3, C=C),
        # "linearsvc": svm.LinearSVC(C=C)
    }

def fit(models, X, Y):
    result = {}
    for key, model in models.items():
        result[key] = model.fit(X, Y)
    return result


def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

if __name__ == '__main__':
    data = prepare_train_data('btc')
    train_data, test_data = train_test_split(data, 0.05)
    models = build_modle()
    X_train = []
    Y_train = np.zeros(shape=(1,0))
    X_test = []
    Y_test = np.zeros(shape=(1,0))
    for (line_number, (index, d)) in enumerate(train_data.iterrows()):
        if line_number-7 >= 0:
            X_train.append([train_data['diff_1_close_price'][line_number-7:line_number].values, train_data['diff_1_volume'][line_number-7:line_number].values])
            Y_train = np.append(Y_train, d["trend_tag"])

    for (line_number, (index, d)) in enumerate(test_data.iterrows()):
        if line_number-7 >= 0:
            X_test.append([test_data['diff_1_close_price'][line_number-7:line_number].values, test_data['diff_1_volume'][line_number-7:line_number].values])
            Y_test = np.append(Y_test, d["trend_tag"])

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    nsamples, nx, ny = X_train.shape
    X_train = X_train.reshape((nsamples,nx*ny))
    nsamples, nx, ny = X_test.shape
    X_test = X_test.reshape((nsamples,nx*ny))
    print("============================")



    result = fit(models, X_train, Y_train)
    print(Y_test)
    for key, fix_model in result.items():
        y_hat = fix_model.predict(X_test)
        print("============="+key+"===============")
        print(y_hat)



