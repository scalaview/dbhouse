import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pywt   # python 小波变换的包
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA

def wt(data, keyname, wavefunc, level, m, n):
    """
    小波降噪函数
    - level: 分解层数；
    - data: 保存列表类型的字典；
    - keyname: 键名；
    - index_list: 待处理序列；
    - wavefunc: 选取的小波函数；
    - m,n 选择进行阈值处理的小波系数层数
    """
    # 分解
    coeff = pywt.wavedec(data[keyname], wavefunc, mode='sym', level=level)
    # 设置 sgn 函数
    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0
    # 降噪过程
    for i in range(m, n + 1):  # 选取小波系数层数为 m~n 层
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2 * np.log(len(cD)))  # 计算阈值
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # 使用 sgn 函数向零收缩
            else:
                coeff[i][j] = 0  # 低于阈值置零
    # 重新构建
    denoised_index = pywt.waverec(coeff, wavefunc)
    # 为了避免出现负值的情况，取绝对值
    # abs_denoised_list = list(map(lambda x: abs(x), denoised_data_list))
    # 返回降噪结果
    return denoised_index



# 打包为函数
def preTest(data, size, pre_size=1):
    fre_size = pre_size - size

    index_list = np.array(data['closeIndex'])[:size]  # 最后10个数据排除用来做预测
    date_list1 = np.array(data['tradeDate'])[:size]
    predata = data
    for x in range(0, pre_size):
        predata = predata.append({'closeIndex': 0}, ignore_index=True)
    index_for_predict = np.array(predata['closeIndex'])[fre_size:]  # 预测的真实值序列
    date_list2 = np.array(predata['tradeDate'])[fre_size:]

    # 分解
    A2,D2,D1 = pywt.wavedec(index_list,'db4',mode='sym',level=2)  # 分解得到第4层低频部分系数和全部4层高频部分系数
    coeff = [A2,D2,D1]

    # 对每层小波系数求解模型系数
    order_A2 = sm.tsa.arma_order_select_ic(A2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
    order_D2 = sm.tsa.arma_order_select_ic(D2,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q
    order_D1 = sm.tsa.arma_order_select_ic(D1,ic='aic')['aic_min_order']   # AIC准则求解模型阶数p,q

    # 对每层小波系数构建ARMA模型
    # 值得注意的是，有时候用AIC准则求解的模型参数来建模会报错，这时候请调节数据时间长度。
    model_A2 =  ARMA(A2,order=order_A2)   # 建立模型
    model_D2 =  ARMA(D2,order=order_D2)
    model_D1 =  ARMA(D1,order=order_D1)

    results_A2 = model_A2.fit()
    results_D2 = model_D2.fit()
    results_D1 = model_D1.fit()

    A2_all,D2_all,D1_all = pywt.wavedec(np.array(predata['closeIndex']),'db4',mode='sym',level=2) # 对所有序列分解
    delta = [len(A2_all)-len(A2),len(D2_all)-len(D2),len(D1_all)-len(D1)] # 求出差值，则delta序列对应的为每层小波系数ARMA模型需要预测的步数


    # 预测小波系数 包括in-sample的和 out-sample的需要预测的小波系数
    pA2 = model_A2.predict(params=results_A2.params,start=1,end=len(A2)+delta[0])
    pD2 = model_D2.predict(params=results_D2.params,start=1,end=len(D2)+delta[1])
    pD1 = model_D1.predict(params=results_D1.params,start=1,end=len(D1)+delta[2])
    print(len(pA2))
    print(len(pD2))
    print(len(pD1))

    # 重构
    coeff_new = [pA2,pD2,pD1]
    denoised_index = pywt.waverec(coeff_new,'db4')


    # 输出10个预测值
    # temp_data_wt = {'real_value':index_for_predict,'pre_value_wt':denoised_index[fre_size:],'err_wt':denoised_index[fre_size:]-index_for_predict,'err_rate_wt/%':(denoised_index[fre_size:]-index_for_predict)/index_for_predict*100}
    # predict_wt = pd.DataFrame(temp_data_wt,index = date_list2,columns=['real_value','pre_value_wt','err_wt','err_rate_wt/%'])
    # print(predict_wt)
    print(denoised_index[10:])
    print(index_for_predict[10:])
    print("*********************************************************************")

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error
import utils

def prepare_train_data(syml):
    import models
    hist = pd.read_sql_query('SELECT `date` AS `tradeDate`, close_price AS closeIndex FROM daily_prices WHERE fsymbol="'+syml+'" AND tsymbol="USDT"', models.engine)
    return hist.fillna(0)



if __name__ == '__main__':
    preTest(prepare_train_data('btc'), -2, 3)

