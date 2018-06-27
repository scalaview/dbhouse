import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout, LSTM
from sklearn import preprocessing

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

def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


def extract_window_data(df, window_len=7):
    window_data = []
    for (line_number, (index, d)) in enumerate(df.iterrows()):
        if line_number-window_len >= 0:
            tmp = df[line_number-window_len:line_number].copy()
            window_data.append(tmp.values)
    return np.array(window_data)


def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

if __name__ == '__main__':
    np.random.seed(42)
    data = prepare_train_data('btc')
    test_size = 0.02
    window_len = 7

    train_data, test_data = train_test_split(data, test_size)
    test_data = pd.concat([train_data[-window_len:], test_data]) #拼接上前7天的数据
    X_train = train_data[['diff_1_volume', 'diff_1_close_price']]
    Y_train = train_data['trend_tag'].values[window_len:]
    X_test = test_data[['diff_1_volume', 'diff_1_close_price']]
    Y_test = test_data['trend_tag'].values[window_len:]

    # data params
    total = data.shape[0]
    # model params
    lstm_neurons = 200
    epochs = 500
    batch_size = 500
    loss = 'mae'
    dropout = 0.0025
    optimizer = 'adam'
    target_col = 'look_back_1_close_price'
    X_train = extract_window_data(X_train, window_len)
    X_test = extract_window_data(X_test, window_len)

    model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    preds = model.predict(X_test).squeeze()
    print("==========preds============")
    print(list(map(round, preds)))
    print("==========Y_test============")
    print(Y_test)