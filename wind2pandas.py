# -*- coding: utf-8 -*-
"""
__author__ = 'laofish'
__title__=rvdata.py
__mtime__ = 2016-12-18

update:2017-12-24
"""

import pandas as pd
import numpy as np
import datetime
# import tushare as ts
from WindPy import *

import traceback


import matplotlib.pyplot as plt


# import time


def price2ret(price):
    '''已知价格，求对数收益率'''

    # 如果不用np
    # list根本做不了点除

    price = np.array(price)
    # ret=np.log(price[:-1] / price[1:]) #[:-1]是去掉最后一个值
    ret = np.log(price[1:] / price[:-1])  # 前面SB了，居然把这个除数被除数弄错了

    return ret


def getrealizedVol_HF(price):
    '''利用高频数据，计算已实现波动率'''
    ret = price2ret(price)  # 高频价格序列
    # ret=array(ret)
    RV = sum(ret * ret)
    realizedvol = np.sqrt(252 * RV)

    return realizedvol


def getrealizedVol_20day(price):
    '''利用日频数据，计算已实现波动率'''
    ret = price2ret(price)  # 高频价格序列
    # ret=array(ret)
    RV = sum(ret * ret)
    realizedvol = np.sqrt(252 * RV/len(ret))

    return realizedvol

def dateFomortChange(sdate):
    # 这一小段程序的作用是将 前面的时间字符串，转化为wind接受的字符串

    import time #可能要覆盖什么重名的吧

    t = time.strptime(sdate, "%Y-%m-%d")
    y, m, d = t[0:3]
    p = datetime(y, m, d)
    stradingdate = p.strftime('%Y%m%d')

    return stradingdate

###############################################

w.start()
whilecount = 0
while not (w.isconnected()) and (whilecount < 5):
    time.sleep(5)
    w.start()
    whilecount = whilecount + 1

# 数据获取
sdate="2017-01-01"
edate="2017-12-20"

sdate2=sdate+" 09:00:00"
edate2=edate+" 15:38:27"

ulticker="510050.SH"

data = w.wsi(ulticker, "close", sdate2, edate2, "BarSize=5")
dateSeries = w.tdays(sdate, edate, "Period=D")

ivixdata = w.wsd("000188.SH", "close", sdate, edate, "Period=D")

# 加上之前之后20和之后20个交易日的价格数据
uldata=w.wsd(ulticker, "close", sdate, edate, "Period=D")

ulprice=pd.DataFrame(uldata.Data,index=uldata.Fields,columns=uldata.Times)
ulprice=ulprice.T #将矩阵转置

# 后移20个交易日 默认交易日
sdatelag20=w.tdaysoffset(20, sdate, "")
edatelag20=w.tdaysoffset(20, edate, "")
uldata_lag=w.wsd(ulticker, "close", sdatelag20.Data[0][0], edatelag20.Data[0][0], "Period=D")

ulpricelag=pd.DataFrame(uldata_lag.Data,index=uldata_lag.Fields,columns=uldata_lag.Times)
ulpricelag=ulpricelag.T #将矩阵转置


# 换成np和pd
cp = pd.Series(data.Data[0])
ct = pd.Series(data.Times)
cd = pd.Series([x.date() for x in ct])
df = pd.DataFrame({'close': cp, 'wdate': cd})
# ds=pd.Series(dateSeries.Data[0])

# 这个方式更好
# # 演示如何将api返回的数据装入Pandas的DataFrame
# 将ivxi装到df里面去
ivixdf=pd.DataFrame(ivixdata .Data,index=ivixdata .Fields,columns=ivixdata.Times)
ivixdf=ivixdf.T #将矩阵转置


rvS = []  # 已实现波动率
dateSA=[]
try:
    # fdmData=ts.get_stock_basics()
    # print(fdmData)
    for dateS in dateSeries.Times:
        # cp_inday = df[df.wdate == dateS.date()]
        cp_inday = df[df.wdate == dateS]
        rv_inday = getrealizedVol_HF(cp_inday.close)
        rvS.append(rv_inday)
        dateSA.append(dateS)

except:
    # 处理异常
    traceback.print_exc()
    print('debug')

# list转成pd series
rvSpd=pd.Series(rvS)
# rvSpd.to_excel('rvseries.xlsx')


winnum=20 # 20天均线
# mdpd=pd.rolling_mean(rvSpd,ma)
mdpd=rvSpd.rolling(window=winnum,center=False).mean()

# ivix series, wind给的数据 乘了一百
sivix=ivixdf['CLOSE']/100

# 日数据的标准差
b=0.9619 # 20标准差的估计误差
# preulstd=ulprice.rolling(window=winnum,center=False).std()*sqrt(250)/b
# lagulstd=ulpricelag.rolling(window=winnum,center=False).std()*sqrt(250)/b

preulstd= []  # 已实现波动率
# dateSA=[]
try:
    # fdmData=ts.get_stock_basics()
    # print(fdmData)
    for i in range(len(dateSeries.Times)):
        if i<len(dateSeries.Times)-20:
            cp_inday = ulprice[i:i+20]
            rv_inday = getrealizedVol_20day(cp_inday.CLOSE)
            preulstd.append(rv_inday)
        else:
            preulstd.append(np.nan)
        # dateSA.append(dateS)

except:
    # 处理异常
    traceback.print_exc()
    print('debug')
# list转成pd series
# 因为上面，我每个值都是取的从第一个向后20个，但是实际上在第一天我并不知道后面20个交易日的股价是什么
preulstdSpd=pd.Series(preulstd).shift(20)
# rvSpd.to_excel('rvseries.xlsx')


# # data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
# plt.plot(dateSA,rvS,label="$RV$")
plt.plot(dateSA,rvSpd,'y-.',label="$Rv$")
plt.plot(dateSA,mdpd,'r',label="$rv_ma$")
plt.plot(dateSA,sivix,'g',label="$iVIX$")

plt.plot(dateSA,preulstdSpd,'k',label="$ulstd$",linewidth = 2.)

# 这个函数好用
forwardstd=preulstdSpd.shift(-20)
# 移动（超前和滞后）数据
# 移动（shifting）指的是沿着时间轴将数据前移或后移。Series 和 DataFrame 都有一个 .shift() 方法用于执行单纯的移动操作，index 维持不变：
# lang:python
# >>> ts
# 2011-01-01    0.362289
# 2011-01-02    0.586695
# 2011-01-03   -0.154522
# 2011-01-06    0.222958
# dtype: float64
# >>> ts.shift(2)
# 2011-01-01         NaN
# 2011-01-02         NaN
# 2011-01-03    0.362289
# 2011-01-06    0.586695
# dtype: float64
# >>> ts.shift(-2)
# 2011-01-01   -0.154522
# 2011-01-02    0.222958
# 2011-01-03         NaN
# 2011-01-06         NaN
# dtype: float64

plt.plot(dateSA,forwardstd,'m',label="$forwulstd$",linestyle = 'dashed',linewidth = 2.)

# 用时间移动重新去数据并不好，因为可能index并不能对应上
# 这里我还是在原来的序列里面找，把把数据向前提20个值


plt.xlabel('T')
# plt.ylabel('rv')
plt.legend()


# >>> datetime(2016, 12, 19)
# Out[25]:
# datetime.datetime(2016, 12, 19, 0, 0)
# >>> datetime(2016, 12, 19).date()
# Out[26]:
# datetime.date(2016, 12, 19)
# >>> dateSA[134].date()==datetime(2016, 12, 19).date()
# Out[27]:
# True

# 加看空的箭头
# kankongIndex=dateSA.index(datetime(2016, 12, 19, 0, 0, 0, 5000))
#
# plt.annotate('20161218_kankong',
# ha = 'center', va = 'bottom',
# xytext = (dateSA[kankongIndex], 0.2),
# xy = (dateSA[kankongIndex], mdpd[kankongIndex]),
# arrowprops = { 'facecolor' : 'black', 'shrink' : 0.05 })

plt.show() #是一定要这个
plt.savefig('rv_wacht.png')
# sns.tsplot(rvS, time=dateSA)
# plt.show() #是一定要这个
print('end of strategy')
w.stop()


# plt.grid(True, linestyle = "-.", color = "r", linewidth = "1")


# 面向对象的方式(不适合用交互式方式绘图)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(x, x * 2)
# # 使用面向对象的方式显示网格
# ax.grid(True, color="r")
#
# plt.show()

###############################################################

# 有交互效果，可以在交互式编译器中观察每一步的执行过程
#
# plt.plot(x, x * 2)
#
# # True 显示网格
# # linestyle 设置线显示的类型(一共四种)
# # color 设置网格的颜色
# # linewidth 设置网格的宽度
# plt.grid(True, linestyle="-.", color="r", linewidth="3")
#
# plt.show()
