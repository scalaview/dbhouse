import models
import requests
import pandas as pd
import utils
from sqlalchemy.orm import sessionmaker


def btc_histories():
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=BTC&tsym=USD&limit=2000')
    return data_struct(res.json()['Data'])

def eos_histories():
    endpoint = 'https://min-api.cryptocompare.com/data/histoday'
    res = requests.get(endpoint + '?fsym=EOS&tsym=USD&limit=332')
    return data_struct(res.json()['Data'])

def data_struct(data):
    hist = pd.DataFrame(data)
    hist = hist.set_index('time')
    hist.index = pd.to_datetime(hist.index, unit='s')
    hist = hist.join(utils.ROC(hist['close'], 30))
    hist = hist.join(utils.CCI(hist['high'], hist['low'], hist['close'], 30))
    hist = hist.join(utils.EVM(hist['high'], hist['low'], hist['volumeto'], 7, 'EVM_7'))
    hist = hist.join(utils.EVM(hist['high'], hist['low'], hist['volumeto'], 14, 'EVM_14'))
    hist = hist.join(utils.force_index(hist['close'], hist['volumeto'], 1))
    hist = hist.fillna(0)
    return hist

def stoce_daily_price(datas, market, fsymbol, tsymbol):
    DailyPrice = models.DailyPrice
    session = sessionmaker()
    session.configure(bind=models.engine)
    s = session()
    for index, data in datas.iterrows():
        dprice = s.query(DailyPrice).filter(DailyPrice.date == index, DailyPrice.fsymbol == fsymbol, DailyPrice.tsymbol == tsymbol).first()
        if dprice is None:
            dprice = DailyPrice(market=market, fsymbol=fsymbol, tsymbol=tsymbol, date=index)
            s.add(dprice)

        dprice.open_price = data['open']
        dprice.high_price = data['high']
        dprice.low_price = data['low']
        dprice.close_price = data['close']
        dprice.volumefrom = data['volumefrom']
        dprice.volumeto = data['volumeto']
        dprice.evm_7 = data['EVM_7']
        dprice.evm_14 = data['EVM_14']
        dprice.cci_30 = data['CCI']
        dprice.roc_30 = data['ROC']
        dprice.forceIndex_1 = data['ForceIndex']
        s.commit()
    s.close()
    print('finish!')


if __name__ == '__main__':
    # stoce_daily_price(btc_histories(), "ALL", "BTC", "USDT")
    stoce_daily_price(eos_histories(), "ALL", "EOS", "USDT")
