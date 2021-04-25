#!/usr/bin/env python
# encoding: utf-8
"""
@version: ??
@author: laofish
@contact: laofish@outlook.com
@site: http://www.laofish.com
@file: rvdata2.py
@time: 2018-06-30 12:17


update:20201006 重新跑一下

计算高频已实现波动率
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import traceback
# import tushare as ts
from WindPy import *
# 也不知道干嘛的，反正错误提示要用这个
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


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
    '''利用高频数据，计算已实现波动率
	需要留意每次输入都是一个交易日的数据，不能是多个交易日的
	返回值对应的是一个交易日的年化已实现波动率
	'''
    ret = price2ret(price)  # 高频价格序列
    # ret=array(ret)
    RV = sum(ret * ret)
    realizedvol = np.sqrt(252 * RV)

    return realizedvol


def getrealizedVol_20day(price):
    '''利用日数据，计算已实现波动率'''
    ret = price2ret(price)  # 日频价格序列
    # ret=array(ret)
    RV = sum(ret * ret)
    realizedvol = np.sqrt(252 * RV / len(ret))

    return realizedvol


def dateFomortChange(sdate):
    # 这一小段程序的作用是将 前面的时间字符串，转化为wind接受的字符串

    import time  # 可能要覆盖什么重名的吧

    t = time.strptime(sdate, "%Y-%m-%d")
    y, m, d = t[0:3]
    p = datetime(y, m, d)
    stradingdate = p.strftime('%Y%m%d')

    return stradingdate


###############################################
if __name__ == '__main__':

    w.start()
    whilecount = 0
    while not (w.isconnected()) and (whilecount < 5):
        time.sleep(5)
        w.start()
        whilecount = whilecount + 1

    # 数据获取
    sdate = "2020-01-03"
    edate = "2021-04-08"

    sdate2 = sdate + " 09:00:00"
    edate2 = edate + " 15:38:27"

    # ulticker = "600258.SH"
    # ulticker = "002384.SZ"
    # ulticker = "603515.SH"
    # ulticker = "000333.SZ"
    # ulticker = "601231.SH"

    ulticker = "510050.SH"

    data = w.wsi(ulticker, "close", sdate2, edate2, "BarSize=5")
    dateSeries = w.tdays(sdate, edate, "Period=D")

    # uldata = w.wsd(ulticker, "close", sdate, edate, "Period=D")
    uldata = w.wsd(ulticker, "close", sdate, edate, "Period=D;PriceAdj=F")  # 前复权

    ulprice = pd.DataFrame(uldata.Data, index=uldata.Fields, columns=uldata.Times)
    ulprice = ulprice.T  # 将矩阵转置

    # 换成np和pd
    cp = pd.Series(data.Data[0])
    ct = pd.Series(data.Times)
    cd = pd.Series([x.date() for x in ct])
    df = pd.DataFrame({'close': cp, 'wdate': cd})
    # ds=pd.Series(dateSeries.Data[0])

    rvS = []  # 已实现波动率
    dateSA = []

    try:
        # fdmData=ts.get_stock_basics()
        # print(fdmData)
        for dateS in dateSeries.Times:
            # cp_inday = df[df.wdate == dateS.date()]
            cp_inday = df[df.wdate == dateS]
            rv_inday = getrealizedVol_HF(cp_inday.close)
            rvS.append(rv_inday)
            dateSA.append(dateS)

        # list转成pd series
        rvSpd = pd.Series(rvS)
        # rvSpd.to_excel('rvseries.xlsx')

        winnum = 20  # 20天均线
        # mdpd=pd.rolling_mean(rvSpd,ma)
        mdpd = rvSpd.rolling(window=winnum, center=False).mean()

        # 日频数据计算已实现波动率
        # getrealizedVol_20day(ulprice['CLOSE'])
        rvulstd_20 = []
        for i in range(len(dateSeries.Times)):
            if i < len(dateSeries.Times) - 20:
                cp_inday = ulprice['CLOSE'][i:i + 20]
                # rv_inday_20 = getrealizedVol_20day(cp_inday['CLOSE'])
                rv_inday_20 = getrealizedVol_20day(cp_inday)
                rvulstd_20.append(rv_inday_20)

        t = [np.nan] * 20
        t.extend(rvulstd_20)
        rvdaily = pd.Series(t)

        rvulstd_1day = [np.nan]
        volcorf = 16  # 日涨跌数据和波动率转化系数

        for i in range(1, len(dateSeries.Times)):
            rv_inday_1day = np.abs(ulprice.CLOSE[i] / ulprice.CLOSE[i - 1] - 1)
            rvulstd_1day.append(rv_inday_1day * volcorf)

        # 日数据的标准差
        b = 0.9619  # 20标准差的估计误差
        # preulstd=ulprice.rolling(window=winnum,center=False).std()*sqrt(250)/b
        # lagulstd=ulpricelag.rolling(window=winnum,center=False).std()*sqrt(250)/b
        plt.figure(figsize=(10, 6))

        

        plt.plot(dateSA, rvSpd, 'y-.', label="$Rv5min$")  # 五分钟数据计算所得
        plt.plot(dateSA, mdpd, 'r', label="$rv5min_{ma}$")  # 上述值均值
        plt.plot(dateSA, rvdaily, 'c-.', label="$Rv_{C2C_{20}}$")  # 收盘价数据计算所得

        # 剔除异常数据
        se = pd.Series(rvulstd_1day)
        se[se > 0.8] = 0.8
        rvulstd_1day = se.tolist()

        # 20210409 加这图太脏了，不加了
        # plt.plot(dateSA, rvulstd_1day, 'k.', label="$RvInterDay$") # 日涨跌幅数据计算所得

        plt.xlabel('T')
        # plt.ylabel('rv')
        plt.legend()

        # plt.show()  # 是一定要这个
        plt.savefig('rv_wach_' + str(ulticker) + '_' + edate + '.png')
        # sns.tsplot(rvS, time=dateSA)
        # plt.show() #是一定要这个
        print('end of strategy')
        w.stop()


    except:
        # 处理异常
        traceback.print_exc()
        print('debug')
