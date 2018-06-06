from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import utils
import noise

def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))

    model.compile(loss=loss, optimizer=optimizer)
    return model


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18);
    plt.show()


def prepare_train_data(syml):
    import models
    hist = pd.read_sql_query('SELECT `date`, open_price, high_price, low_price, close_price, volumeto, volumefrom FROM daily_prices WHERE fsymbol="'+syml+'" AND tsymbol="USDT"', models.engine)
    hist = hist.set_index('date')
    hist.index = pd.to_datetime(hist.index, unit='s')
    hist = hist.join(pd.Series(noise.wt(hist, 'close_price', 'db4', 4, 2, 4), name='denoised_close_price', index=hist.index))
    hist = hist.join(pd.Series(hist['denoised_close_price'].shift(-3), name='look_back_1_close_price'))
    return hist.fillna(0)

def run_train(hist):
    np.random.seed(42)
    # data params
    total = hist.shape[0]
    window_len = 7
    test_size = 0.02
    zero_base = True
    # model params
    lstm_neurons = 200
    epochs = 500
    batch_size = 500
    loss = 'mae'
    dropout = 0.0025
    optimizer = 'adam'
    target_col = 'look_back_1_close_price'
    train, test, X_train, X_test, y_train, y_test = utils.prepare_data(hist[['open_price', 'high_price', 'low_price', 'volumeto', 'volumefrom', 'denoised_close_price', 'look_back_1_close_price']], target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
    for x, y, z in np.argwhere(np.isnan(X_train)):
        X_train[(x, y, z)] = 0
    model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    preds = model.predict(X_test).squeeze()
    targets = hist['close_price'][total-preds.shape[0]:]
    maer = mean_absolute_error(preds, y_test)
    print("mean_absolute_error: %f" % maer)
    preds = test[target_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=(targets.index + pd.DateOffset(3)), data=preds)
    line_plot(targets, preds, 'actual', 'prediction', lw=3)
    print(hist['close_price'].tail())
    print(preds.tail(13))
    n_points = 7
    line_plot(targets[-n_points:], preds[-n_points:], 'actual', 'prediction', lw=3)

if __name__ == '__main__':
    run_train(prepare_train_data('BTC'))


