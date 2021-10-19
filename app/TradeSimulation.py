# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 21:37:12 2021

@author: docs9
"""

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
import glob
import calendar

TIME = 'time'
OPEN = 'open'
HIGH = 'high'
LOW = 'low'
CLOSE = 'close'

THRESHOLD = 'threshold'
DELAY = 'delay'
LOSSCUT = 'loscut'

def priceRange(ohlc):
    p = []
    for o, h, l, c in ohlc:
        p.append(c)
    return (max(p), min(p))

NOTHING = 0
LONG = 1
SHORT = -1
class Position():
    def __init__(self, losscut, threshold, delay_minutes):
        self.threshold = threshold
        self.losscut = losscut 
        self.status = NOTHING
        self.delay_minutes = delay_minutes
        self.peak = None
        self.spread = 0.0
        
    def long(self, index, time, open, close):
        self.status = LONG
        self.open_time = time
        self.open_index = index
        self.close_time = None
        self.open_price = close
        self.close_price = None
        self.peak = close
        self.peak_time = time
        self.time_limit = time + timedelta(minutes=self.delay_minutes)
        self.losscut_price = close - (close - open ) * self.losscut
        
    def short(self, index, time, open, close):
        self.status = SHORT
        self.open_index = index
        self.open_time = time
        self.close_time = None
        self.open_price = close
        self.close_price = None
        self.peak = close
        self.peak_time = time
        self.time_limit = time + timedelta(minutes=self.delay_minutes)
        self.losscut_price = close - (close - open) * self.losscut
        
    def update(self, index, time, open, high, low, close):
        if self.status == NOTHING:
            r = (close - open) / open * 100.0
            if r >= self.threshold[0] and r <= self.threshold[1]:
                if r > 0:
                    self.long(index, time, open, close)
                else:
                    self.short(index, time, open, close)
            return None
               
        if self.status == LONG:
            if low <= self.losscut_price:
                return self.square(time, low)
            elif close > self.peak:
                self.peak = close
                self.peak_time = time
        elif self.status == SHORT:
            if high >= self.losscut_price:
                return self.square(time, high)
            elif close < self.peak:
                self.peak = close
                self.peak_time = time
        
        if time >= self.time_limit:
            return self.square(time, close)
                

        
    def square(self, time, price):
        self.close_time = time
        self.close_price = price
        self.profit = self.close_price - self.open_price
        r = self.result()
        self.status = NOTHING
        return r
        
    def result(self):            
        return [self.status, self.open_time, self.open_price, self.close_time, self.close_price, self.profit, self.peak_time, self.peak, self.peak - self.open_price]

        
INACTIVE = 0 
ACTIVE_BEGIN = 1
ACTIVE = 2
ACTIVE_END = 3
    
class Simulation:
    def __init__(self, dic, timeframe, trade_time_range):
        time = dic[TIME]
        open = dic[OPEN]
        high = dic[HIGH]
        low = dic[LOW]
        close = dic[CLOSE]
        self.begin_hour = trade_time_range[0][0]
        self.begin_minutes = trade_time_range[0][1]
        self.end_hour = trade_time_range[1][0]
        self.end_minutes = trade_time_range[1][1]
        self.time = time
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.timeframe = timeframe
        self.length = len(time)


    def runLongBar(self, param):
        threshold = param[THRESHOLD]
        delay = param[DELAY]
        losscut = param[LOSSCUT]
        vmin = np.min(threshold)
        vmax = np.max(threshold)
        threshold_arrange = [vmin, vmax]
        position = Position(losscut, threshold_arrange, delay)
        trades = []
        
        tnow = self.time[0]
        tbegin = datetime(tnow.year, tnow.month, tnow.day, self.begin_hour, self.begin_minutes)
        tend = datetime(tnow.year, tnow.month, tnow.day, self.end_hour, self.end_minutes)
        if tbegin > tend:
            tend += timedelta(days=1)
        
        for i in range(self.length):
            t = self.time[i]
            if t < tbegin:
                continue
            if t > tend:
                break
            
            trade = position.update(i, t, self.open[i], self.high[i], self.low[i], self.close[i])
            if trade is not None:
                trades.append(trade)
        if position.status != NOTHING:
            trade = position.square(self.time[-1], self.close[-1])
            trades.append(trade)
            
        profit = 0.0
        for trade in trades:
            profit += trade[5]
        return profit, trades