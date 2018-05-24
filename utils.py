import pandas as pd
from pandas_datareader import data as web
from datetime import datetime

# Force Index
def force_index(data_close, data_volume, ndays, syml='ForceIndex'):
    return pd.Series(data_close.diff(ndays) * data_volume / 100000, name=syml)

# Rate of Change (ROC)
def ROC(data_close, n, syml='ROC'):
    N = data_close.diff(n)
    D = data_close.shift(n)
    return pd.Series(N/D, name=syml)


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
