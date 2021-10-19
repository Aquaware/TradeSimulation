# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../common'))

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from Timeframe import Timeframe
from CandlePlot import CandlePlot, BandPlot, makeFig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import calendar
from ta.trend import SMAIndicator

import TradeSimulation

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


THRESHOLD = 'threshold'
DELAY = 'delay'
LOSSCUT = 'loscut'

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
    

def timeFilter(tohlc_dic, year, month, day, hourmins):
    time = tohlc_dic[TIME]
    try:
        t1 = datetime(year, month, day, hourmins[0][0], hourmins[0][1])
        t2 = datetime(year, month, day, hourmins[1][0], hourmins[1][1])
        if t1 > t2:
            t2 += timedelta(days=1)
        if t1 > time[-1] or t2 < time[0]:
            return (0, None)
    except:
        return (0, None)
    
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
    if begin is None or stop is None:
        return (0, None)

    dic = {}
    dic[TIME] = time[begin:stop]
    dic[OPEN] = tohlc_dic[OPEN][begin: stop]
    dic[HIGH] = tohlc_dic[HIGH][begin: stop]
    dic[LOW] = tohlc_dic[LOW][begin: stop]
    dic[CLOSE] = tohlc_dic[CLOSE][begin: stop]
    return (stop - begin, dic)


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

def drawGraph(market, title, timeframe, tohlc, display_time_range, trades):
    fig = plt.figure(figsize=(14, 6)) 
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
            
    for trade in trades:
        [status, topen, open_price, tclose, close_price, profit1, tpeak, peak_price, profit2] = trade
        if status > 0:
            color = 'green'
            marker = '^'
        else:
            color = 'red'
            marker = 'v'
        graph1.drawMarker(topen, open_price, marker, color, markersize=10)
        graph1.drawMarker(tpeak, peak_price, 'x', color, markersize=10)
            
            
    flag = []
    for o, c in zip(open, close):
        try:
            v = (c - o) / o * 100.0
        except:
            v = 0
        flag.append(v)
    
    graph2 = BandPlot(fig, ax2, 'Flag')
    graph2.xlimit(t_range)
    graph2.drawLine(time, flag, timerange=t_range)
    #ax1.legend()
    #ax2.legend()

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

def showByDay(market, year, month, day, timeframe, display_time_range, tohlc_dic):
    title = market + ' ' + str(year) + '-' + str(month) + '-' + str(day) + ' (' + weekday(year, month, day)[:3] + ') ' 
    
    count, dic = timeFilter(tohlc_dic, year, month, day, [[8, 0], [7, 0]])
    if count > 100:
        drawGraph(market, title, timeframe, dic, display_time_range)

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
        try:
            t = datetime(year, month, day)
        except:
            break
        if t < tend:
            days.append(day)
        else:
            break
    return days
    
def show(market, timeframe, year, month):
    display_time_range = [[8, 0], [7, 0]]    
    tohlc = importClickSec("../click_sec_data", market, year, month)
    tohlc_dic = separate(tohlc, timeframe)
    
    for day in dayRange(year, month):
        showByDay(market, year, month, day, timeframe, display_time_range, tohlc_dic)
  
    
def filterUpper(array, threshold):
    out = []
    for v in array:
        if v > threshold:
            out.append(v)
    return out

def filterLower(array, threshold):
    out = []
    for v in array:
        if v < threshold:
            out.append(v)
    return out
    
def rangeHistogram(market, timeframe):
    for year in [2019, 2020, 2021]:
        ranges = []
        for month in range(1, 13):
            tohlc = importClickSec("../click_sec_data", market, year, month)
            tohlc_dic = separate(tohlc, timeframe)
            for (o, c) in zip(tohlc_dic[OPEN], tohlc_dic[CLOSE]):
                ranges.append((c - o) / o * 100)
                
        higher = filterUpper(ranges, 0.3)
        lower = filterLower(ranges, -0.3)
        vmin = np.min(ranges)
        vmax = np.max(ranges)
        
        fig, axes= plt.subplots(1,2)
        axes[1].hist(higher, bins=10)        
        #axes[1].set_title(market + "-" + timeframe.symbol + " " +  str(year) + "  Min: " + str(vmin) + "  Max: " + str(vmax))        
        axes[0].hist(lower, bins=10)        
        axes[0].set_title(market + "-" + timeframe.symbol + " " +  str(year) + "  Min: " + str(vmin)[:7] + "  Max: " + str(vmax)[:7])
        fig.show()
        
def judge(rng, threshold):
    lower = np.min(threshold)
    upper = np.max(threshold)
    if rng > lower and rng < upper:
        return True
    else:
        return False
       
def longBarLowerStrategy(market, timeframe, threshold, after_minutes):
    out = []
    for year in [2019, 2020, 2021]:
        longBars = []
        tend = None
        values = []
        for month in range(1, 13):
            tohlc = importClickSec("../click_sec_data", market, year, month)
            tohlc_dic = separate(tohlc, timeframe)    
            for (t, o, c) in zip(tohlc_dic[TIME], tohlc_dic[OPEN], tohlc_dic[CLOSE]):
                rng = (c - o) / o * 100
                if tend is None:
                    if judge(rng, threshold):
                        tend = t + timedelta(minutes= after_minutes)
                        values.append([t, o, c])
                else:
                    if t > tend:
                        longBars.append(values)
                        tend = None
                        values = []
                    else:
                        values.append([t, o, c])
        
        print('*** Year ', year)
        for longBar in longBars:
            closes = []
            begin = None
            for i, (t, o, c) in enumerate(longBar):
                if i == 0:
                    begin = [t, c]
                    print('Begin: t: ', t, 'Range: ', (c - o) / o * 100, 'Close:', c)
            
                else:
                    closes.append(c)
                    if i == len(longBar)-1:
                        end = [t, c]
                        print(' -> End: t: ', t, 'close: ', c, 'Profit:', c - begin[1])
            
            if len(closes) > 0:
                minv = np.min(closes)
                maxv = np.max(closes)
                is_short = threshold[0] < 0
                if is_short:
                    profit = minv - begin[1]
                else:
                    profit = maxv - begin[1]
                print (' -> Min: ', minv, maxv, 'profit: ', profit)
                out.append([year, begin[0], begin[1], end[0], end[1], end[1] - begin[1], minv, maxv, profit])
            print ('')
            
    data = []
    s = 0.0
    for d in out:
        s += d[-1]
        dd = d
        dd.append(s)
        data.append(dd)
    
    df = pd.DataFrame(data=data, columns=['Year', 'tbegin', 'close', 'tend', 'close', 'profit', 'close-min', 'close-max', 'profit', 'profit-sum'])
    #df.to_excel(market + 'LongBarStrategy.xlsx', index=False)
    
    return s
    
    
def test1():
    market = "JP225"
    timeframe = Timeframe("M10")
    year = 2021
    month = 10
    show(market, timeframe, year, month)
    
def analyze():
    market = "SPOT_GOLD" #"CHNA50" #"US30" #WTI" #SPOT_GOLD" #"JP225"
    tf = "M5"
    timeframe = Timeframe(tf)
    #rangeHistogram(market, timeframe)
    
    r1 =  [[0.3, 0.6], [0.4, 0.6], [0.5, 0.7], [-0.3, -0.6], [-0.4, -0.6], [-0.5, -0.7]]
    r2 =  [[0.5, 1.0], [1.0, 2.0], [3.0, 5.0], [-0.5, -1.0], [-1.0, -2.0], [-3.0, -5.0]]
    
    out = []
    for threshold in r1:
        for delay in [15, 30, 60, 90, 120]:
            profit = longBarLowerStrategy(market, timeframe, threshold, delay)
            out.append([market, tf, threshold, delay, profit])

    df = pd.DataFrame(data= out, columns=['Market', 'Timeframe', 'threshold', 'delay', 'profit'])
    df.to_excel('./docs/' + market + '-LongBarStragegySummary.xlsx', index=False)    
    
    
def trade():
    market = "SPOT_GOLD" #"JP225" #"CHNA50" #"US30" #WTI" #SPOT_GOLD" #"JP225"
    tf = "M15"
    timeframe = Timeframe(tf)
    data_time_range = [[8, 0], [7, 0]]
    params =  [ {THRESHOLD: [0.25, 0.5], DELAY: 60,  LOSSCUT: 0.5},
                {THRESHOLD: [-0.25, -0.5], DELAY: 60,  LOSSCUT: 0.5}]
    out = []
    for year in [2019]: #, 2020, 2021]:
        for month in range(1, 13):
            tohlc = importClickSec("../click_sec_data", market, year, month)
            tohlc_dic = separate(tohlc, timeframe)
            for day in dayRange(year, month):
                date_str = str(year) + '-' + str(month) + '-' + str(day)
                count, dic = timeFilter(tohlc_dic, year, month, day, data_time_range)
                if count > 50:
                    sim = TradeSimulation.Simulation(dic, timeframe, data_time_range)
                    trades = []
                    for param in params:
                        profit, trade = sim.runLongBar(param)
                        if len(trade) > 0:
                            out += trade
                            trades += trade
                    title = market + " " + date_str                    
                    drawGraph(market, title, timeframe, dic, data_time_range, trades)
    #df = pd.DataFrame(data=out, columns=['Status', 'OpenTime', 'OpenPrice', 'CloseTime', 'ClosePrice', 'Profit1', 'MaxTime', 'MaxPrice', 'profit2'])
    #df.to_excel('./docs/' + market + '-tradeSummary.xlsx', index=False)
                    
if __name__ == '__main__':
    trade()
    

    
