from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import utils
import noise
from statsmodels.tsa.stattools import coint, adfuller


# 幸运地是对两个序列协整性的检验已经有现成的函数，通过stattools包的coint函数可直接检验协整性。
# from statsmodels.tsa.stattools import coint
# coint(X1, X2)

def prepare_train_data(syml):
    import models
    hist = pd.read_sql_query('SELECT `date`, open_price, high_price, low_price, close_price, volumeto, volumefrom FROM daily_prices WHERE fsymbol="'+syml+'" AND tsymbol="USDT" AND date >= "2018-01-01 00:00:00"', models.engine)
    hist = hist.set_index('date')
    hist.index = pd.to_datetime(hist.index, unit='s')
    return hist.fillna(0)

def check_for_stationarity(X, cutoff=0.01):
    # 原假设H0:单位根存在（非平稳）
    # 我们通过重要的p值，以确认自己这个序列是平稳的
    pvalue = adfuller(X)[1]
    if pvalue < cutoff:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely stationary.')
        return True
    else:
        print('p-value = ' + str(pvalue) + ' The series ' + X.name +' is likely non-stationary.')
        return False


if __name__ == '__main__':
    data = prepare_train_data('BTC')
    diff_close = data['close_price'].diff(1)[1:]
    check_for_stationarity(diff_close)
    diff_close.plot()
    plt.show()