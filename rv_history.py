# -*- coding: utf-8 -*-
"""
__author__ = 'laofish'
__title__=rv_history.py
__mtime__ = 2017-08-12
"""

import pandas as pd
import numpy as np
import datetime

from WindPy import *
# from optbkfun import *
import traceback

# import statsmodels.api as sm
# from sklearn import linear_model

import matplotlib.pyplot as plt

import time


def price2ret(price):
    '''已知价格，求对数收益率'''

    # 如果不用np
    # list根本做不了点除

    price = np.array(price)
    # ret=np.log(price[:-1] / price[1:]) #[:-1]是去掉最后一个值
    ret = np.log(price[1:] / price[:-1])  # 前面SB了，居然把这个除数被除数弄错了

    return ret


def getrealizedVol_20day(price):
    '''利用高频数据，计算已实现波动率'''
    ret = price2ret(price)  # 高频价格序列
    # ret=array(ret)
    RV = sum(ret * ret)
    realizedvol = np.sqrt(252 * RV/len(ret))

    return realizedvol


sns.set_style("dark")
w.start()
whilecount = 0
while not (w.isconnected()) and (whilecount < 5):
    time.sleep(5)
    w.start()
    whilecount = whilecount + 1

# 数据获取
sdate="2016-01-01"
edate="2017-12-21"

sdate2=sdate+" 09:00:00"
edate2=edate+" 15:38:27"

ulticker="510050.SH"

dateSeries = w.tdays(sdate, edate, "Period=D")


ivixdata = w.wsd("000188.SH", "close", sdate, edate, "Period=D")

# 加上之前之后20和之后20个交易日的价格数据
uldata=w.wsd(ulticker, "close", sdate, edate, "Period=D")

ulprice=pd.DataFrame(uldata.Data,index=uldata.Fields,columns=uldata.Times)
ulprice=ulprice.T #将矩阵转置


# 这个方式更好
# # 演示如何将api返回的数据装入Pandas的DataFrame
# 将ivxi装到df里面去
ivixdf=pd.DataFrame(ivixdata .Data,index=ivixdata .Fields,columns=ivixdata.Times)
ivixdf=ivixdf.T #将矩阵转置


sivix=ivixdf['CLOSE']/100

# 日数据的标准差
b=0.9619 # 20标准差的估计误差
# preulstd=ulprice.rolling(window=winnum,center=False).std()*sqrt(250)/b
# lagulstd=ulpricelag.rolling(window=winnum,center=False).std()*sqrt(250)/b

rvulstd= []  # 已实现波动率
# dateSA=[]
try:
    # fdmData=ts.get_stock_basics()
    # print(fdmData)
    for i in range(len(dateSeries.Times)):
        if i<len(dateSeries.Times)-20:
            cp_inday = ulprice[i:i+20]
            rv_inday = getrealizedVol_20day(cp_inday.CLOSE)
            rvulstd.append(rv_inday)
        else:
            pass
        # dateSA.append(dateS)

except:
    # 处理异常
    traceback.print_exc()
    print('debug')

rvulstd_60= []
try:
    # fdmData=ts.get_stock_basics()
    # print(fdmData)
    for i in range(len(dateSeries.Times)):
        if i<len(dateSeries.Times)-60:
            cp_inday = ulprice[i:i+60]
            rv_inday_60 = getrealizedVol_20day(cp_inday.CLOSE)
            rvulstd_60.append(rv_inday_60)
        else:
            pass
        # dateSA.append(dateS)

except:
    # 处理异常
    traceback.print_exc()
    print('debug')

# list转成pd series
rvSpd=pd.Series(rvulstd[0:-40])
rvSpd.name='c2c_20_rv'

rvSpd60=pd.Series(rvulstd_60)
rvSpd60.name='c2c_60_rv'

df=pd.DataFrame(rvSpd)
# df['ivix']=sivix
# 少20个点
df['ivix']=sivix[0:-60].tolist()

df['c2c_60_rv']=rvSpd60


df.to_excel('ivxi_His_rv_2.xlsx')
pass
