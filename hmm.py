from hmmlearn.hmm import GaussianHMM
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import pyplot


def prepare_train_data(syml):
    import models
    hist = pd.read_sql_query('SELECT `date`, open_price, high_price AS high, low_price AS low, close_price AS close, volumeto AS volume FROM daily_prices WHERE fsymbol="'+syml+'" AND tsymbol="USDT" AND date between "2018-01-01" and "2018-06-24" ', models.engine)
    return hist

def origin_hmm():
    df = prepare_train_data('btc')
    # 数据准备
    close = df['close']
    high = df['high'][5:]
    low = df['low'][5:]
    volume = df['volume'][5:]
    money = df['volume'][5:]
    datelist = df['date'][5:]
    # 计算当日对数收益率，五日对数收益率，当日对数高低价差
    logreturn = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[4:]
    logreturn5 = np.log(np.array(close[5:]))-np.log(np.array(close[:-5]))
    diffreturn = (np.log(np.array(high))-np.log(np.array(low)))
    closeidx = close[5:]
    # 训练并完成预测
    X = np.column_stack([logreturn,diffreturn,logreturn5])
    hmm = GaussianHMM(n_components = 6, covariance_type='diag',n_iter = 5000).fit(X)
    latent_states_sequence = hmm.predict(X)
    # 绘制市场状态序列
    sns.set_style('white')
    plt.figure(figsize = (15, 8))
    for i in range(hmm.n_components):
        state = (latent_states_sequence == i)
        plt.plot(datelist[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
        plt.legend()
        plt.grid(1)
    plt.show()

def my_hmm():
    from sklearn import preprocessing
    min_max_scaler = preprocessing.MinMaxScaler()
    df = prepare_train_data('btc')
    # 数据准备
    close = df['close']
    high = df['high'][5:]
    low = df['low'][5:]
    volume = df['volume'][5:]
    money = df['volume'][5:]
    datelist = df['date'][1:]
    # 计算当日对数收益率，五日对数收益率，当日对数高低价差
    logreturn = (np.log(np.array(close[1:]))-np.log(np.array(close[:-1])))[4:]
    logreturn5 = np.log(np.array(close[5:]))-np.log(np.array(close[:-5]))
    diffreturn = (np.log(np.array(high))-np.log(np.array(low)))
    closeidx = close[1:]
    diff_1_close_price = df["close"].diff(1).shift(-1)
    diff_2_close_price = df["close"].diff(2).shift(-2)
    diff_1_close_price = (diff_1_close_price/df["close"] * 100)
    diff_2_close_price = (diff_2_close_price/df["close"] * 100)
    diff_1_close_price = min_max_scaler.fit_transform(diff_1_close_price.values[:-1].reshape(-1, 1))

    diff_1_data = df["volume"].diff(1).shift(-1)
    diff_2_data = df["volume"].diff(2).shift(-2)
    diff_1_data = (diff_1_data/df["volume"] * 100)
    diff_2_data = (diff_2_data/df["volume"] * 100)
    diff_1_data = min_max_scaler.fit_transform(diff_1_data.values[:-1].reshape(-1, 1))
    # 训练并完成预测
    X = np.column_stack([diff_1_close_price, diff_1_data],)
    hmm = GaussianHMM(n_components = 6, covariance_type='diag',n_iter = 5000).fit(X)
    latent_states_sequence = hmm.predict(X)
    # 绘制市场状态序列
    sns.set_style('white')
    plt.figure(figsize = (15, 8))
    for i in range(hmm.n_components):
        state = (latent_states_sequence == i)
        plt.plot(datelist[state],closeidx[state],'.',label = 'latent state %d'%i,lw = 1)
        plt.legend()
        plt.grid(1)
    plt.show()

if __name__ == '__main__':
    # my_hmm()
    origin_hmm()