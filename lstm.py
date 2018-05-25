from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import utils

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

if __name__ == '__main__':
    import models
    hist = pd.read_sql_query('SELECT `date`, open_price, high_price, low_price, close_price, volumeto FROM daily_prices WHERE id > 3827 AND fsymbol="EOS" AND tsymbol="USDT"', models.engine)
    hist = hist.set_index('date')
    hist.index = pd.to_datetime(hist.index, unit='s')
    hist = hist.join(pd.Series(hist['close_price'].shift(-1), name='look_back_1_close_price'))
    hist = hist.fillna(0)
    np.random.seed(42)
    # data params
    window_len = 7
    test_size = 0.1
    zero_base = True
    # model params
    lstm_neurons = 20
    epochs = 50
    batch_size = 4
    loss = 'mae'
    dropout = 0.25
    optimizer = 'adam'
    target_col = 'look_back_1_close_price'
    train, test, X_train, X_test, y_train, y_test = utils.prepare_data(hist, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)
    for x, y, z in np.argwhere(np.isnan(X_train)):
        X_train[(x, y, z)] = 0
    model = build_lstm_model(X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss, optimizer=optimizer)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)
    targets = test['close_price'][window_len:]
    preds = model.predict(X_test).squeeze()
    maer = mean_absolute_error(preds, y_test)
    print("mean_absolute_error: %f" % maer)
    preds = test[target_col].values[:-window_len] * (preds + 1)
    preds = pd.Series(index=targets.index, data=preds)
    line_plot(targets, preds, 'actual', 'prediction', lw=3)
    print(preds.tail())
    n_points = 7
    line_plot(targets[-n_points:], preds[-n_points:], 'actual', 'prediction', lw=3)

