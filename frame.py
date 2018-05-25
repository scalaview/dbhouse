import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import utils
import models


def btc_frame(syml="Btc"):
    hist = pd.read_sql_query('SELECT `date`, open_price, high_price, low_price, close_price, volumeto, evm_7, evm_14, cci_30, roc_30, forceIndex_1 FROM daily_prices WHERE fsymbol="BTC" AND tsymbol="USDT" AND `date` > "2018-01-01"', models.engine)
    hist = hist.set_index('date')
    hist.index = pd.to_datetime(hist.index, unit='s')
    hist.plot(y=['close_price', 'evm_7'], subplots = 1, title=syml, figsize=(24, 12))
    hist.plot(y=['close_price', 'evm_14'], subplots = 1, title=syml, figsize=(24, 12))
    hist.plot(y=['close_price', 'cci_30'], subplots = 1, title=syml, figsize=(24, 12))
    hist.plot(y=['close_price', 'roc_30'], subplots = 1, title=syml, figsize=(24, 12))
    hist.plot(y=['close_price', 'forceIndex_1'], subplots = 1, title=syml, figsize=(24, 12))
    plt.show(block=True)

def eos_frame(syml="EOS"):
    hist = pd.read_sql_query('SELECT `date`, open_price, high_price, low_price, close_price, volumeto, evm_7, evm_14, cci_30, roc_30, forceIndex_1 FROM daily_prices WHERE fsymbol="EOS" AND tsymbol="USDT" AND `date` > "2018-01-01"', models.engine)
    hist = hist.set_index('date')
    hist.index = pd.to_datetime(hist.index, unit='s')
    hist.plot(y=['close_price', 'evm_7'], subplots = 1, title=syml, figsize=(24, 12))
    hist.plot(y=['close_price', 'evm_14'], subplots = 1, title=syml, figsize=(24, 12))
    hist.plot(y=['close_price', 'cci_30'], subplots = 1, title=syml, figsize=(24, 12))
    hist.plot(y=['close_price', 'roc_30'], subplots = 1, title=syml, figsize=(24, 12))
    hist.plot(y=['close_price', 'forceIndex_1'], subplots = 1, title=syml, figsize=(24, 12))
    plt.show(block=True)



if __name__ == '__main__':
    btc_frame()
    eos_frame()
