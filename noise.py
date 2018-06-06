import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pywt   # python 小波变换的包


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
