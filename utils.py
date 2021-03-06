import pandas as pd
from pandas_datareader import data as web
from datetime import datetime
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
# Force Index
def force_index(data_close, data_volume, ndays, syml='ForceIndex'):
    return pd.Series(data_close.diff(ndays) * data_volume / 100000, name=syml)

# Rate of Change (ROC)
def ROC(data_close, n, syml='ROC'):
    N = data_close.diff(n)
    D = data_close.shift(n)
    return pd.Series((N/D).replace([np.inf, -np.inf], 0), name=syml)


# Commodity Channel Index
def CCI(data_high, data_low, data_close, ndays, syml='CCI'):
    TP = (data_high + data_low + data_close) / 3
    return pd.Series((TP - pd.Series(TP).rolling(ndays).mean()) / (0.015 * pd.Series(TP).rolling(ndays).std()), name=syml)


# Ease of Movement
def EVM(data_high, data_low, data_volume, ndays, syml='EVM'):
  dm = ((data_high + data_low)/2) - ((data_high.shift(1) + data_low.shift(1))/2)
  br = (data_volume / 100000000) / ((data_high - data_low))
  EVM = dm / br
  return pd.Series(pd.Series(EVM).rolling(ndays, center=False).mean(), name=syml)



def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry. """
    return df / df.iloc[0] - 1


def normalise_min_max(df):
    """ Normalise dataframe column-wise min/max. """
    return (df - df.min()) / (data.max() - df.min())

def extract_window_data(df, window_len=10, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of len `window_data`.

        :param window_len: Size of window
        :param zero_base: If True, the data in each window is normalised to reflect changes
            with respect to the first entry in the window (which is then always 0)
    """
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i: (i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df, test_size=test_size)
    # extract window data
    X_train = extract_window_data(train_data.drop(columns=[target_col]), window_len, zero_base)
    X_test = extract_window_data(test_data.drop(columns=[target_col]), window_len, zero_base)
    # extract targets
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1

    return train_data, test_data, X_train, X_test, y_train, y_test


def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18);
    plt.show()



def scatter(x, y, label1=None, label2=None, title=''):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    #设置标题
    ax.set_title(title)
    #设置X轴标签
    plt.xlabel(label1)
    #设置Y轴标签
    plt.ylabel(label2)
    #画散点图
    ax.scatter(x,y, c='r', marker='o')
    #设置图标
    plt.legend('x1')
    #显示所画的图
    plt.show()

def standardization():
    from sklearn import preprocessing
    import numpy as np
    X = np.array([[ 1., -1.,  2.],
                   [ 2.,  0.,  0.],
                   [ 0.,  1., -1.]])
    X_scaled = preprocessing.scale(X)
    X_scaled
