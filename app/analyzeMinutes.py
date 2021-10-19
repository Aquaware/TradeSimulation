# -*- coding: utf-8 -*-

import os
import sys
current_dir = os.path.abspath(os.path.dirname(__file__))

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from Timeframe import Timeframe
from CandlePlot import CandlePlot, BandPlot, makeFig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import calendar
from ta.trend import SMAIndicator

TIME = 'time'
OPEN = 'open'
HIGH = 'high'
LOW = 'low'
CLOSE = 'close'
DIF = 'close-open'
SIGMA = 'sigma'
DMA_SLOW = 'dma_slow'
DMA_FAST = 'dma_fast'
BID = 'bid'
ASK = 'ask'
MID = 'mid'
SPREAD = 'spread'

def weekday(year, month, day):
    d = date(year, month, day)
    day_index = d.weekday()
    return calendar.day_name[day_index][:3]

def dirPath(root, stock, year):
    path = root + stock + '/' + str(year).zfill(4) + '/' 
    return path

def filename(stock, year, month, day):
    path = stock + '_Tick_' + str(year).zfill(4) + '-' + str(month).zfill(2) + '-' + str(day).zfill(2) + '.csv'
    return path
    

def timeFilter(time, year, month, day, hourmins):
    try:
        t1 = datetime(year, month, day, hourmins[0][0], hourmins[0][1])
        t2 = datetime(year, month, day, hourmins[1][0], hourmins[1][1])
    except:
        return (None, None)
    
    if t1 > t2:
        t2 += timedelta(days=1)

    try:
        if t1 > time[-1] or t2 < time[0]:
            return (None, None)
    except:
        return (None, None)
    
    begin = None
    stop = None
    for (i, t) in enumerate(time):
        if begin is None:
            if t >= t1:
                begin = i
        else:
            if t >= t2:
                stop = i
                break
    if stop is None:
        stop = len(time)
    return (begin, stop)


def candle(prices):
    o = prices[0]
    c = prices[-1]
    h = np.max(prices)
    l = np.min(prices)
    return [o, h, l, c]

def tohlc2ohlc(tohlc):
    time = []
    ohlc = []
    for value in tohlc:
        time.append(value[0])
        ohlc.append(value[1:])
    return (time, ohlc)

def separate(tohlc, timeframe):
    time = None
    data = []
    open = None
    high = None
    low = None
    close = None
    for t, o, h, l, c in tohlc:
        tt = timeframe.roundTime(t)
        if time is None:
            time = tt
            open = o
            high = h
            low = l
            close = c
        else:
            if tt > time:
                data.append([time, open, high, low, close])
                time = tt
                open = o
                high = h
                low = l
                close = c
            else:
                if h > high:
                    high = h
                if l < low:
                    low = l
                close = c
    data.append([time, open, high, low, close])
    
    time = []
    open = []
    high = []
    low = []
    close = []
    dif = []
    for t, o, h, l, c in data:
        if t is None or c is None or o is None:
            continue
        time.append(t)
        open.append(o)
        high.append(h)
        low.append(l)
        close.append(c)
        dif.append(c - o)
        
    dic = {}
    dic[TIME] = time
    dic[OPEN] = open
    dic[HIGH] = high
    dic[LOW] = low
    dic[CLOSE] = close
    dic[DIF] = dif
    return dic

def SMA(array, window):
    ind = SMAIndicator(close=pd.Series(array), window=window)
    close = ind.sma_indicator()
    return close.values.tolist()

def drawGraph(market, title, timeframe, tohlc, display_time_range):
    fig = plt.figure(figsize=(18, 12)) 
    gs = gridspec.GridSpec(6, 1)   #縦,横
    ax1 = plt.subplot(gs[0:5, 0])
    ax2 = plt.subplot(gs[5:6, 0])
    
    time = tohlc[TIME]
    open = tohlc[OPEN]
    high = tohlc[HIGH]
    low = tohlc[LOW]
    close = tohlc[CLOSE]
    ohlc = []
    for o, h, l, c in zip(open, high, low, close):
        ohlc.append([o, h, l, c])
    
    if display_time_range is None:
        t_range = (time[0], time[-1])
    else:
        tt = time[0]
        t0 = datetime(tt.year, tt.month, tt.day, display_time_range[0][0], display_time_range[0][1])
        t1 = datetime(tt.year, tt.month, tt.day, display_time_range[1][0], display_time_range[1][1])
        if t1 < t0:
            t1 += timedelta(days=1)
        t_range = [t0, t1]
        
    graph1 = CandlePlot(fig, ax1, title)
    graph1.xlimit(t_range)
    graph1.drawCandle(time, ohlc, timerange=t_range)
    
    windows =[3, 5, 7, 10, 20]
    colors = ['salmon', 'red', 'gray', 'yellowgreen', 'green']
    mas = {}
    for w, color in zip(windows, colors):
        ma = SMA(close, w)
        mas['MA' + str(w)] = ma
        if color is not None:
            graph1.drawLine(time, ma, color=color, label='MA' + str(w))
            
    flag = []
    for o, c in zip(open, close):
        try:
            v = (c - o) / o
        except:
            v = 0
        flag.append(v)
    
    graph2 = BandPlot(fig, ax2, 'Flag')
    graph2.xlimit(t_range)
    graph2.drawLine(time, flag, timerange=t_range)
    ax1.legend()
    ax2.legend()

def priceRange(ohlc):
    p = []
    for o, h, l, c in ohlc:
        p.append(c)
    return (max(p), min(p))

def drawByDay(market, tf, ticks, year, month):
    for day in range(1, 32):
        count, data = timeFilter(ticks, year, month, day, [[21, 30], [4, 30]])
        #ticks =  fromDb(market, year, month, day, [22, 23], None)
        if count > 100:
            (tohlc, spreads) = ticks2TOHLC(tf, data)
            time, ohlc = tohlc2ohlc(tohlc)
            price_range = priceRange(ohlc)
            title = market + ' (' + tf.symbol + ')   ' + str(year) + '-' + str(month) + '-' + str(day) + ' (' + weekday(year, month, day)[:3]+ ')  Price Range: ' + str(price_range[0] - price_range[1]) 
            drawGraph(title, tf, time, ohlc)   

def showByDay(market, year, month, day, timeframe, display_time_range, data):
    title = market + ' ' + str(year) + '-' + str(month) + '-' + str(day) + ' (' + weekday(year, month, day)[:3] + ') ' 
    drawGraph(market, title, timeframe, data, display_time_range)

def importClickSec(dir_path, market, year, month):
    ym = str(year) + str(month).zfill(2)
    dir_path = dir_path + '/' + market + '/' + market + '_' +  ym + '/' + ym + '/'
    tohlc = []
    for day in range(1, 32):
        file_name = market +  '_' + ym + str(day).zfill(2) +  '.csv'
        path = os.path.join(dir_path, file_name)
        try:    
            df0 = pd.read_csv(path, encoding='sjis')
            df = df0[['日時', '始値(BID)', '高値(BID)', '安値(BID)', '終値(BID)']]
            values= df.values.tolist()
        except:
            continue
    
        for value in values:
            [tint, o, h, l, c] = value
            t = str(tint)
            time = datetime(int(t[0:4]), int(t[4:6]), int(t[6:8]), int(t[8:10]), int(t[10:12]))
            tohlc.append([time, float(value[1]), float(value[2]), float(value[3]), float(value[4])])
    return tohlc
    
def dayRange(year, month):
    if month == 12:
        tend = datetime(year + 1, 1, 1)
    else:
        tend = datetime(year, month + 1, 1)
        
    days = []
    for day in range(1, 32):
        t = datetime(year, month, day)
        if t < tend:
            days.append(day)
        else:
            break
    return days
    
def show():
    market = "US30"
    timeframe = Timeframe("M5")
    year = 2021
    month = 10
    display_time_range = [[8, 0], [7, 0]]    
    tohlc = importClickSec("../click_sec_data", market, year, month)
    data = separate(tohlc, timeframe)
    
    for day in dayRange(year, month):
        showByDay(market, year, month, day, timeframe, display_time_range, data)
  
    
if __name__ == '__main__':
    show()
    

