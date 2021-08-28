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
import glob
import calendar
from ta.trend import SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands

import statistics

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
    
def fromDb(stock, year, month, day, hours, save_dir_path):
    t1 = datetime(year, month, day, hours[0], 0)
    t2 = datetime(year, month, day, hours[1], 0)
    if hours[1] < hours[0]:
        t2 += timedelta(days=1)
    db = PriceDatabase()
    data = db.tickRange(stock, t1, t2, value_filter=[20000, 35000])
    
    if save_dir_path is not None:
        filepath = dirPath(save_dir_path, stock, year) + filename(stock, year, month, day)
        df = pd.DataFrame(data=data, columns=['Time', 'Bid', 'Ask', 'Mid', 'Volume'])
        df.to_csv(filepath, index=False)
    return data

def fromCsv(stock, year, month):
    file_path = '../tickdata/' + stock + '_' + str(year) + '_' + str(month).zfill(2) + '.csv'
    if not os.path.exists(file_path):
        return []
    df = pd.read_csv(file_path)
    out = []
    count = 0
    for (index, t, b, a, m, v) in df.values:
        try:
            time = datetime.strptime(t, '%Y-%m-%d %H:%M:%S.%f')
        except:
            try:
                time = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            except:
                continue
        if b > 0 and a > 0:
            count += 1
        out.append([time, b, a, m, v])
    return count, out


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
 
def timeFilterTicks(ticks, year, month, day, hourmins):
    try:
        t1 = datetime(year, month, day, hourmins[0][0], hourmins[0][1])
        t2 = datetime(year, month, day, hourmins[1][0], hourmins[1][1])
    except:
        return (None, None)
    
    if t1 > t2:
        t2 += timedelta(days=1)

    time = []
    bids = []
    asks = []
    mids = []
    spreads = []
    for tick in ticks:
        t, bid, ask, mid, volume = tick
        if t >= t1 and t <= t2:
            time.append(t)
            bids.append(bid)
            asks.append(ask)
            mids.append(mid)
            spreads.append(bid - ask)
    d = {TIME: time, BID: bids, ASK: asks,  MID: mids, SPREAD: spreads}
    return d

def candle(prices):
    o = prices[0]
    c = prices[-1]
    h = np.max(prices)
    l = np.min(prices)
    return [o, h, l, c]


# tick_list : [[time, bid, ask, mid, volume], [...]]
def ticks2TOHLC(timeframe: Timeframe, tick_list):
    current = None
    mids = []
    spread = []
    spreads = []
    tohlc = []
    for tick in tick_list:
        [time, bid, ask, mid, volume] = tick
        t = timeframe.roundTime(time)
        if current is None:
            current = t
            mids.append(mid)
            spread.append(ask - bid)
        else:
            if t > current:
                ohlc = candle(mids)
                tohlc.append([current] + ohlc)
                spreads.append(np.mean(spread))
                current = t
                mids = [mid]
                spread = [ask - bid]
            else:
                mids.append(mid)
                spread.append(ask - bid)
    return (tohlc, spreads)

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


def DMA(array, window, timeframe):
    ma = SMA(array, window)
    if timeframe.symbol == 'S10':
        k = 6.0
    elif timeframe.symbol == 'M1':
        k  = 1.0
    elif timeframe.symbol == 'M5':
        k = 0.2
    elif timeframe.symbol == 'M15':
        k = 1 / 15
        
    dma = delta(ma, k)
    return dma
    
def STDEV(array, window):
    out = []
    for i in range(len(array)):
        if i < window:
            out.append(None)
        else:
            d = array[i - window + 1: i + 1]
            std = statistics.stdev(d)
            out.append(std)
    return out
    
def BB(array, window, sigma):
    ind = BollingerBands (close=pd.Series(array), window=window, window_dev=sigma)
    upper = ind.bollinger_hband()
    lower = ind.bollinger_lband()
    return (upper.values.tolist(), lower.values.tolist())  


def ATR(data, window):
    high = data[HIGH]
    low = data[LOW]
    close = data[CLOSE]
    ind = AverageTrueRange (high=pd.Series(high), low=pd.Series(low), close=pd.Series(close), window=window)
    a = ind.average_true_range()
    return a.values.tolist()


def SIGMA(data, term, window):
    high = data[HIGH]
    low = data[LOW]
    
    sigma = []
    n = len(high)
    begin = n - window
    if begin < 0:
        return []
    
    for i in range(n):
        if i < window:
            sigma.append(None)
            continue
        
        d = []
        for j in range(window):
            try:
                maxv = max(high[i - window + j: i - window + j + term]) 
                minv = min(low[i - window + j: i - window + j + term])
                d.append(maxv - minv)
            except:
                print(i, j, i - j + term)
        s = 3 * statistics.stdev(d)
        sigma.append(s)
    return sigma

def SigmaRate(price, window, mulval):
    std = STDEV(price, window)
    ma = SMA(price, window)
    rate = []
    for p, d, m in zip(price, std, ma):
        if p is None or d is None or m is None:
            rate.append(None)
        else:
            if d == 0.0:
                rate.append(0)
            else:
                rate.append((p - m ) / d * mulval)
    return rate

def maSigmaBand(data, ma_fast, ma_slow, upper, lower):
    open = data[OPEN]
    high = data[HIGH]
    low = data[LOW]
    dif = data[DIF]
    out = []
    for o, h, l, d, fast, slow, up, low in zip(open, high, low, dif, ma_fast, ma_slow, upper, lower):
        if np.isnan(fast) or np.isnan(slow) or np.isnan(up) or  np.isnan(low):
            out.append(0)
            continue
        if d == 0:
            out.append(0)
        elif d > 0:
            w = up - fast
            if (o > fast - w * 0.2) and (h > fast + w * 0.6):
                out.append(1)
            else:
                out.append(0)
        else:
            w = fast - low
            if (o  < fast + w * 0.2) and (l < fast - w * 0.6):
                out.append(2)
            else:
                out.append(0)

    return out

def maBand(mas):
    ma5 = mas['MA5']
    ma20 = mas['MA20']
    ma60 = mas['MA60']
    ma100 = mas['MA100']
    ma200 = mas['MA200']
    
    out = []
    for m5, m20, m60, m100, m200 in zip(ma5, ma20, ma60, ma100, ma200):
        if m5 >= m20 and m20 > m60 and m60 > m100:
            status = 1
        elif m20 >= m60 and m60 > m100:
            status = 2
        elif m60 > m100:
            status = 3
        elif m5 <= m20 and m20 < m60 and m60 < m100:
            status = 4
        elif m60 <= m20 and m100 < m60:
            status = 5
        elif m60 < m100:
            status = 6
        else:
            status = 0
        out.append(status)

    colors = ['black', '#0000ff', '#4444aa', '#444488', '#ff0000', '#aa4444', '#884444']    
    return out, colors
    

def delta(data, mulval):
    out = []
    for i in range(0, len(data)):
        if i == 0:
            out.append(None)
        elif data[i] is None or data[i-1] is None:
            out.append(None)
        else:
            out.append( (data[i] - data[i-1]) * mulval)
    return out
    
def drawGraph(market, title, timeframe, tohlc, ticks, trades, display_time_range):
    try:
        atr = ATR(tohlc, 15)
    except:
        return
    
    
    fig = plt.figure(figsize=(20, 12)) 
    gs = gridspec.GridSpec(30, 1)   #縦,横
    ax1 = plt.subplot(gs[0:16, 0])
    ax2 = plt.subplot(gs[16:21, 0])
    ax3 = plt.subplot(gs[21:26, 0])
    ax4 = plt.subplot(gs[26:31, 0])
    
    time = tohlc[TIME]
    open = tohlc[OPEN]
    high = tohlc[HIGH]
    low = tohlc[LOW]
    close = tohlc[CLOSE]
    ohlc = []
    for o, h, l, c in zip(open, high, low, close):
        ohlc.append([o, h, l, c])
    
    if timeframe.symbol == 'S10':
        begin = 300
    elif timeframe.symbol == 'M1':
        begin = 30
            
    graph1 = CandlePlot(fig, ax1, title)
    graph1.xlimit(display_time_range)
    graph1.drawCandle(time, ohlc, timerange=display_time_range)
    
    graph3 = CandlePlot(fig, ax3, 'SigmaBand')
    graph2 = CandlePlot(fig, ax2, 'DMA')


    windows =[5, 15, 60, 120]
    colors = [None, 'red', 'blue', 'orange']
    mas = {}
    for w, color in zip(windows, colors):
        ma = SMA(close, w)
        mas['MA' + str(w)] = ma
        if color is not None:
            graph1.drawLine(time, ma, color=color, label='MA' + str(w))
            
    crosses = drawCrossing(graph1, time, mas)
            
            
            
    #upper1, lower1 = BB(close, 12, 1.0)
    #upper2, lower2 = BB(close, 12, 2.0)
    #graph1.drawLine(time, upper1, color='green', linestyle='dashed', linewidth=1.0, label='MA12+sigma')
    #graph1.drawLine(time, lower1, color='green', linestyle='dashed', linewidth=1.0, label='MA12-sigma')
    #graph1.drawLine(time, upper2, color='green', linewidth=1.0, label='MA12+2sigma')
    #graph1.drawLine(time, lower2, color='green', linewidth=1.0, label='MA12-2sigma')
    
    if trades is not None:    
        for trade in trades:
            status, open_time, close_time, open_price, close_price, profit = trade
            if status == LONG:
                color = 'green'
                marker = '^'
            elif status == SHORT:
                color = 'red'
                marker = 'v'
            graph1.drawMarker(open_time, open_price, marker, color)
            graph1.drawMarker(close_time, close_price, '*', color)
    

    
    graph4 = CandlePlot(fig, ax4, 'ATR w=15')
    graph4.xlimit(display_time_range)
    
    dji_range = np.arange(10, 40, 10)
    
    if timeframe.symbol == 'M1':
        gold_range = np.arange(0.2, 11.0, 0.2)
    elif timeframe.symbol == 'M5':
        gold_range = np.arange(0.5, 11.0, 0.5)
        
    if market == 'dji':
        rng = dji_range
    elif market == 'gold':
        rng = gold_range
    for v in rng:
        graph4.hline(v, color='lightgray')
    
    if timeframe.symbol == 'S10':
        limit = 20
    elif timeframe.symbol == 'M1':
        limit = 50
        
    if market == 'dji':
        limit = 50
    elif market == 'gold':
        limit  = 1.0

    if timeframe.symbol == 'M5':
        limit *= 5.0
    graph4.drawLine(time, atr, color='blue', ylim=(0, limit), label='ATR w=5')
    
    if ticks is not None:
        drawTicks(graph1, graph2, ticks, display_time_range)
        return
    
    drawDMA(graph2, market, timeframe, time, display_time_range, mas)
    drawCrosses2(graph2, crosses)
    
    drawSigmaBand(graph3, time, display_time_range, close)
    drawCrosses2(graph3, crosses)
    
    ax1.legend()
    ax2.legend()
    ax3.legend()
    ax4.legend()
    
    
def crossOver(time, ma_fast, ma_mid, ma_slow):
    cross_over = []
    golden_cross = []
    cross_under = []
    dead_cross = []
    OVER = 1
    UNDER = -1
    for i in range(len(ma_fast)):
        if i == 0:
            continue
        else:
            if ma_fast[i - 1] < ma_mid[ i - 1] and ma_fast[i] >= ma_mid[i]:
                if ma_mid[i] >= ma_slow[i]:
                    golden_cross.append([i, time[i], ma_mid[i]])                                
                else:
                    cross_over.append([i, time[i], ma_mid[i]])  
            elif ma_fast[i - 1] > ma_mid[i - 1] and ma_fast[i] <= ma_mid[i]:
                if ma_mid[i] <= ma_slow[i]:
                    dead_cross.append([i, time[i], ma_mid[i]])  
                else:
                    cross_under.append([i, time[i], ma_mid[i]])  
                
    return (golden_cross, dead_cross, cross_over, cross_under)
    
    
    
def drawCrossing(graph, time, mas):
    ma_fast = mas['MA15']
    ma_mid = mas['MA60']
    ma_slow = mas['MA120']
    crosses = crossOver(time, ma_fast, ma_mid, ma_slow)
    drawCrosses(graph, crosses)
    return crosses
    
def drawCrosses(graph, crosses):
    (golden_cross, dead_cross, cross_over, cross_under) = crosses
    
    for i, t, v in golden_cross:
        graph.vline(t, color='orange', linewidth=2)
        
    for i, t, v in dead_cross:
        graph.vline(t, color='gray', linewidth=2)
        
    for i, t, v in cross_over:
        graph.drawMarker(t, v + 1, 'X', 'Cyan', markersize=5)
        
    for i, t, v in cross_under:
        graph.drawMarker(t, v - 1, 'X', 'Red', markersize=5)
    
def drawCrosses2(graph, crosses):
    (golden_cross, dead_cross, cross_over, cross_under) = crosses
    
    for i, t, v in golden_cross:
        graph.vline(t, color='orange', linewidth=2)
        
    for i, t, v in dead_cross:
        graph.vline(t, color='gray', linewidth=2)
        
    for i, t, v in cross_over:
        graph.vline(t, color='orange', linewidth=1)
        
    for i, t, v in cross_under:
        graph.vline(t, color='gray', linewidth=1)
        
    
    

def drawSigmaBand(graph, time, trange, price):
    sigma1 = SigmaRate(price,60, 1)
    sigma2 = SigmaRate(price,15, 1)
    graph.xlimit(trange)
    graph.drawBar(time, sigma1, ylim=(-4, 4), label='w=60')
    graph.drawLine(time, sigma2, color='blue', label='w=15/60')
    graph.hline(-1.0, color='lightgray')
    graph.hline(1.0, color='lightgray')
    graph.hline(0.0, color='lightgray', linewidth=2.0)
    graph.hline(2.0, color='lightgray')
    graph.hline(-2.0, color='lightgray')
    graph.ax.set_xticks([])

        
    
    #graph.drawLine(time, sigma2, color='red', ylim=(-15, 15), label=label1)
    #graph.drawLine(time, dif3, color='orange', ylim=(-15, 15), label=label3)
    #graph.drawBar(time, dif2, ylim=(-15, 15), label=label3)
    return 


def drawDMA(graph, market, timeframe, time, trange, mas):
    if timeframe.symbol == 'S10':
        label1 = 'dMA60'
        label2 = 'dMA100'
        label3 = 'dMA200'
        dif1 = delta(mas['MA60'] , 6.0)
        dif2 = delta(mas['MA100'], 6.0)
        dif3 = delta(mas['MA200'], 6.0)
    else:
        label1 = 'dMA5'
        label2 = 'dMA15'
        label3 = 'dMA60'
        if timeframe.symbol == 'M1':
            c = 1.0
        elif timeframe.symbol == 'M5':
            c = 1.0 / 5.0
        dif1 = delta(mas['MA5'] , c)
        dif2 = delta(mas['MA15'], c)
        dif3 = delta(mas['MA60'], c)
        
    #graph2 = CandlePlot(fig, ax2, 'deltaMA')
    
    if market == 'dji':
        lim = (-15, 15)
    elif market == 'gold':
        if timeframe.symbol == 'M1':
            lim = (-0.5, 0.5)
        elif timeframe.symbol == 'M5':
            lim = (-0.2, 0.2)
            
    graph.xlimit(trange)
    graph.drawLine(time, dif1, color='blue', ylim=lim, label=label1)
    graph.drawLine(time, dif3, color='orange', ylim=lim, label=label3)
    graph.drawBar(time, dif2, ylim=lim, label=label3)
    graph.ax.set_xticks([])
    return 
    
    
def drawTicks(graph1, graph2, ticks, trange):
    ttick = ticks[TIME]
    bid = ticks[BID]    
    graph.drawLine(ttick, bid, color='blue', label='Tick')
    ma = SMA(bid, 30)
    dlt = delta(ma, 1.0)
    #spread = ticks[SPREAD]
    #graph = CandlePlot(fig, ax2, 'deltaTick')
    graph.xlimit(trange)
    graph.drawBar(ttick, dlt, ylim=(-1.5, 1.5), label='delta tick')
    #graph.drawLine(ttick, spread, color='red', ylim=(-20, 20), label='spread')
    return
    
    
    
def priceRange(ohlc):
    p = []
    for o, h, l, c in ohlc:
        p.append(c)
    return (max(p), min(p))

NOTHING = 0
LONG = 1
SHORT = -1
class Position:
    def __init__(self, loss_cut, threshold, delay):
        self.threshold = threshold
        self.loss_cut = loss_cut 
        self.status = NOTHING
        self.delay = delay
        self.peak = None
        self.spread = 0.0
        
    def long(self, index, time, price):
        self.status = LONG
        self.open_time = time
        self.open_index = index
        self.close_time = None
        self.open_price = price
        self.close_price = None
        self.peak = price
        
    def short(self, index, time, price):
        self.status = SHORT
        self.open_index = index
        self.open_time = time
        self.close_time = None
        self.open_price = price
        self.close_price = None
        self.peak = price
        
    def update(self, index, time, price, dma_fast, dma_slow):
        if self.status == NOTHING:
            if dma_slow > self.threshold and dma_fast > dma_slow:
                self.long(index, time, price)
            elif dma_slow < -self.threshold and dma_fast < dma_slow:
                self.short(index, time, price)
            return None
        
        should_close = False
        elapsed = index - self.open_index
        if elapsed < self.delay:
            return None
        
        k = 1.0
        if self.status == LONG:
            profit = price - self.open_price - self.spread
            if price > self.peak:
                self.peak = price
                
            if dma_fast <= dma_slow * k:
                should_close = True
            
        elif self.status == SHORT:
            profit = self.open_price - price - self.spread
            if price < self.peak:
                self.peak = price 
            if dma_fast >= dma_slow * k:
                should_close = True                
       
        if profit <= - self.loss_cut:
            should_close = True
            
        if should_close:
            return self.square(time, price)
        else:
            return None
        
    def square(self, time, price):
        self.close_time = time
        self.close_price = price
        if self.status == LONG:
            self.profit = self.close_price - self.open_price
        elif self.status == SHORT:
            self.profit = self.open_price - self.close_price
        r = self.result()
        self.status = NOTHING
        return r
        
    def result(self):            
        return [self.status, self.open_time, self.close_time, self.open_price, self.close_price, self.profit]

        
INACTIVE = 0 
ACTIVE_BEGIN = 1
ACTIVE = 2
ACTIVE_END = 3
    
class Simulation:
    def __init__(self, dic, timeframe, trade_time_range):
        time = dic[TIME]
        close = dic[CLOSE]
        dma_slow = dic[DMA_SLOW]
        dma_fast = dic[DMA_FAST]
        sigma = dic[SIGMA]
        
        t = time[0]
        self.trade_begin = datetime(t.year, t.month, t.day, trade_time_range[0][0], trade_time_range[0][1])
        self.trade_end = datetime(t.year, t.month, t.day, trade_time_range[1][0], trade_time_range[1][1])
        if self.trade_end < self.trade_begin:
            self.trade_end += timedelta(days=1)
        
        self.time = time
        self.close = close
        self.dma_slow = dma_slow
        self.dma_fast = dma_fast
        self.timeframe = timeframe
        self.ma_window_slow = 5
        self.ma_window_fast= 12
        self.length = len(time)
        self.sigma = sigma

    
    def run(self, loscut, set_threshold, delay):
        position = Position(loscut, set_threshold, delay)
        trades = []
        for i in range(self.length):
            t = self.time[i]
            if t < self.trade_begin:
                continue
            if t > self.trade_end:
                break
            
            price = self.close[i]
            trade = position.update(i, t, price, self.dma_fast[i], self.dma_slow[i])
            if trade is not None:
                trades.append(trade)
        if position.status != NOTHING:
            trade = position.square(self.time[-1], self.close[-1])
            trades.append(trade)
            
        profit = 0.0
        for trade in trades:
            profit += trade[5]
        return profit, trades
    
    
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

    
def main(market, timeframe, year, month, tohlc=None):
    if tohlc is None:
        tf = Timeframe(timeframe)
        _, ticks = fromCsv(market, year, month)
        tohlc, spreads = ticks2TOHLC(tf, ticks)
    data = separate(tohlc)
    dma_slow = DMA(data[CLOSE], 12, timeframe)
    data[DMA_SLOW] = dma_slow
    dma_fast = DMA(data[CLOSE], 5, timeframe)
    data[DMA_FAST] = dma_fast
    sigma = SIGMA(data, 5, 20)
    data[SIGMA] = sigma
    
    trange = [[22, 30], [4, 30]]
    out = []
    total = 0.0

    print(market + ' ' +  str(year) +'-' + str(month) )
    profits = []
    for loss_cut in np.arange(5, 100.0, 5.0):
        for threshold in np.arange(0.0, 5.0, 1):
            for delay in range(1, 3):
                total = 0.0
                for day in range(1, 32):
                    param = {'loss_cut': loss_cut, 'threshold': threshold, 'delay': delay}
                    profit, trades = dayTrade(year, month, day, timeframe, trange, data, param)
                    total += profit
                profits.append([loss_cut, threshold, delay, total])
                if total > 0:
                    print('Losscut: ', loss_cut, ' Threshold: ', threshold, 'Delay: ', delay,  '... Profit: ', total)    
                #df = pd.DataFrame(data=out, columns=['open_time', 'close_time', 'open_price', 'close_price', 'profit'])
                #df.to_excel(market + '-' + str(year) + '-' + str(month) + '_trades.xlsx', index=False)

    df = pd.DataFrame(data=profits, columns=['Loss_cut', 'threshold', 'delay', 'profit'])
    #df.to_excel(market + '-' + str(year) + '-' + str(month) + '_profits.xlsx', index=False)
    
def dayTrade(market, year, month, day, timeframe, display_time_range, trade_time_range, data, ticks, param):
    time = data[TIME]
    open = data[OPEN]
    high = data[HIGH]
    low  = data[LOW]
    close = data[CLOSE]
    dma_slow = data[DMA_SLOW]
    dma_fast = data[DMA_FAST]
    sigma = data[SIGMA]
    dif = data[DIF]
    
    loss_cut = param['loss_cut']
    threshold = param['threshold']
    delay = param['delay']
    
    trades = []
    total = 0.0
    begin, end = timeFilter(time, year, month, day, trade_time_range)
    if ticks is not None:
        tick_data = timeFilterTicks(ticks, year, month, day, trade_time_range)
    else:
        tick_data = None
    if begin is not None and end is not None:
        t = time[begin:end]
        o = open[begin:end]
        h = high[begin:end]
        l = low[begin:end]
        c = close[begin:end]
        s = sigma[begin:end]
        slow = dma_slow[begin:end]
        fast = dma_fast[begin:end]
        dif = dif[begin:end]
        if len(t) < 1:
            return (0.0, None)
        
        tt = t[0]
        t0 = datetime(tt.year, tt.month, tt.day, display_time_range[0][0], display_time_range[0][1])
        t1 = datetime(tt.year, tt.month, tt.day, display_time_range[1][0], display_time_range[1][1])
        if t1 < t0:
            t1 += timedelta(days=1)
        trange = [t0, t1]
        d = {TIME:t, OPEN:o, HIGH:h, LOW: l, CLOSE:c, SIGMA:s, DMA_FAST: fast, DIF: dif, DMA_SLOW: slow}
        #sim = Simulation(d, timeframe, trade_time_range)
        #profit, trade = sim.run(loss_cut, threshold, delay)
        title = market + ' ' + str(year) + '-' + str(month) + '-' + str(day) + ' (' + weekday(year, month, day)[:3] + ')  Range: ' + '{:.1f}'.format(max(c) - min(c))
        drawGraph(market, title, timeframe, d, tick_data, None, trange) #trade)

        return (0.0, None) 
    else:
        return (0.0, None)
    
    
def export(csv_path, market, timeframe, year, month):
    tf = Timeframe(timeframe)
    _, ticks = fromCsv(market, year, month)
    tohlc, spreads = ticks2TOHLC(tf, ticks)
    df = pd.DataFrame(data = tohlc, columns=['time', 'open', 'high', 'low', 'close'])
    df.to_csv(csv_path, index=False)
    
def importIGTickCsv(market, year, month):
    _, ticks = fromCsv(market, year, month)
    return ticks


def importClickSec(dir_path, brand, year, month):
    ym = str(year) + str(month).zfill(2)
    dir_path = dir_path + '/' + brand + '/' + ym + '/'
    tohlc = []
    for day in range(1, 32):
        file_name = brand +  '_' + ym + str(day).zfill(2) +  '.csv'
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
    
def importCsv(csv_path):
    df = pd.read_csv(csv_path)
    values = df.values.tolist()
    out = []
    for value in values:
        t = datetime.strptime(value[0], '%Y-%m-%d %H:%M:%S')
        o = value[1]
        h = value[2]
        l = value[3]
        c = value[4]
        out.append([t, o, h, l, c])
    return out
    
def test(market, tf, year, month, tohlc, ticks):
    timeframe = Timeframe(tf)
    data = separate(tohlc, timeframe)
    dma_slow = DMA(data[CLOSE], 12, timeframe)
    data[DMA_SLOW] = dma_slow
    dma_fast = DMA(data[CLOSE], 5, timeframe)
    data[DMA_FAST] = dma_fast
    sigma = SIGMA(data, 5, 20)
    data[SIGMA] = sigma
    
    #dji
    r = [[[21, 0], [3, 0]]]
    r1 =  [[22, 30], [22, 40]]
    r2 =  [[22, 40], [22, 50]]
    r3 =  [[22,  50], [23, 0]]
    r4 =  [[23, 0], [23, 10]]
    r5 =  [[23, 10], [23, 20]]
    r6 =  [[23, 20], [23, 30]]
    
    
    r = [[[16, 30], [2, 30]]]
    
    display_ranges = r #[r1, r2, r3, r4, r5, r6]
    trade_range = [[8, 0], [5, 0]] 

    loss_cut = 20
    threshold = 0.2
    delay = 2
    
    param = {'loss_cut': loss_cut, 'threshold': threshold, 'delay': delay}
    for day in range(1, 31):
        for display_range in display_ranges:
            profit, trades = dayTrade(market, year, month, day, timeframe, display_range, trade_range, data, ticks, param)
            #df = pd.DataFrame(data=trades, columns=['long/short', 'open_time', 'close_time', 'open_price', 'close_price', 'profit'])        
            #path = market + '_' + str(year) + '-' + str(month) + '-' + str(day) + '_profit.xlsx'
            #df.to_excel(path, index=False)
    
if __name__ == '__main__':
    #export('./data/dji_10sec_2021-08.csv', 'dji', 'S10', 2021, 8)
    #tohlc = importCsv('./data/dji_10sec_2021-08.csv')
    year = 2021
    for month in range(8, 13):
        tohlc = importClickSec('../click_sec_data',  'SPOT_GOLD', year, month)
        #ticks = importIGTickCsv('dji', 2021, 8)
        test('gold', 'M1', year, month, tohlc, None)
    
    

