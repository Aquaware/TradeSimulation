# -*- coding: utf-8 -*-
import os
import sys
#sys.path.append(os.path.join(os.path.dirname(__file__), '../XM'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
import matplotlib.dates as mdates
from datetime import datetime 

import matplotlib.animation as animation



DATE_FORMAT_TIME = '%H:%M'
DATE_FORMAT_DAY = '%m-%d'
DATE_FORMAT_DAY_TIME = '%m-%d %H:%M'

def makeFig(rows, cols, size):
    fig, ax = plt.subplots(rows, cols, figsize=(size[0], size[1]))
    return (fig, ax)

def awarePytime2naive(time):
    naive = datetime(time.year, time.month, time.day, time.hour, time.minute, time.second)
    return naive

def awarePyTime2Float(time):
    naive = awarePytime2naive(time)
    t = mdates.date2num([naive])
    return t[0]

def awarePyTimeList2Float(aware_pytime_list):
    naives = []
    for time in aware_pytime_list:
        naive = awarePytime2naive(time)
        naives.append(naive)
    return mdates.date2num(naives)
    
class CandleGraphic:
    def __init__(self, py_time, ohlc, box_width):
        GREEN = '#00aa88'
        RED = '#ff6666'
        self.box_width = box_width
        self.line_width = 1.0
        self.alpha = 0.7
        self.color_positive = GREEN
        self.box_line_color_positive = 'green'
        self.color_negative = RED
        self.box_line_color_negative = 'red'
        
        t = awarePyTime2Float(py_time)
        open = ohlc[0]
        high = ohlc[1]
        low = ohlc[2]
        close = ohlc[3]
        if close >= open:
            color = self.color_positive
            line_color = self.color_positive
            box_low = open
            box_high = close
            height = close - open
        else:
            color = self.color_negative
            line_color = self.color_negative
            box_low = close
            box_high = open
            height = open - close
            
        line_upper = Line2D(xdata=(t, t),
                            ydata=(box_high, high),
                            color=line_color,
                            linewidth=self.line_width,
                            antialiased=True)
        line_lower = Line2D(xdata=(t, t),
                            ydata=(box_low, low),
                            color=line_color,
                            linewidth=self.line_width,
                            antialiased=True)

        rect = Rectangle(xy=(t - self.box_width / 2, box_low),
                         width=self.box_width,
                         height=height,
                         facecolor=color,
                         edgecolor=color)
        rect.set_alpha(self.alpha)

        self.line_upper = line_upper
        self.line_lower = line_lower
        self.rect = rect
        return
    
    def setObject(self, ax):
        ax.add_line(self.line_upper)
        ax.add_line(self.line_lower)
        ax.add_patch(self.rect)
        return
    
class BoxGraphic:
    def __init__(self, py_time, box_width, value, color):
        self.alpha = 0.7
        t = awarePyTime2Float(py_time)
        if value >= 0:
            box_low = 0.0
            box_height = value
        else:
            box_low = value
            box_height = -value
        rect = Rectangle(xy=(t - box_width / 2, box_low),
                         width=box_width,
                         height=box_height,
                         facecolor=color,
                         edgecolor=color)
        rect.set_alpha(self.alpha)
        self.rect = rect
        return
    
    def setObject(self, ax):
        ax.add_patch(self.rect)
        return
# -----
    
class CandlePlot:
    def __init__(self, fig, ax, title, date_format=DATE_FORMAT_TIME):
        self.fix = fig
        self.ax = ax
        self.title = title
        self.ax.grid()
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        pass
        
    def drawCandle(self, time, ohlc, bar_width=None, timerange=None):    
        self.ax.set_title(self.title)
        n = len(time)
        if bar_width is None:
            v1 = awarePyTime2Float(time[0])
            v2 = awarePyTime2Float(time[-1])
            self.box_width = (v2 - v1 ) / n / 2.0
        else:
            self.box_width = bar_width
            
        self.graphic_objects = []
        begin = None
        end = None
        vmin = None
        vmax = None
        for i in range(n):
            t = time[i]
            if timerange is not None:
                if begin is None:
                    if t >= timerange[0]:
                        begin = i
                        vmin = ohlc[i][2]
                        vmax = ohlc[i][1]
                else:
                    if t <= timerange[1]:
                        if vmin > ohlc[i][2]:
                            vmin = ohlc[i][2]
                        if vmax < ohlc[i][1]:
                            vmax = ohlc[i][1]
                    else:
                        end = i - 1
            value = ohlc[i]
            obj = CandleGraphic(t, value, self.box_width)
            obj.setObject(self.ax)
            self.graphic_objects.append(obj)
            
        if vmin is not None and vmax is not None:
            dw = (vmax - vmin) * 0.4
            self.ylimit([vmin - dw, vmax + dw])
            
        self.ax.autoscale_view()
        return
    
    def drawLine(self, time, value, color='red', linestyle='solid', linewidth=1.0, ylim=None, label=''):
        self.ax.plot(time, value, color=color, linestyle=linestyle, linewidth=linewidth, label=label)
        if ylim is not None:
            self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.grid()
        return
    
    def hline(self, y, color='black', linewidth=1.0):
        xmin, xmax = self.ax.get_xlim()
        self.ax.hlines(y=y, color=color, linewidth=linewidth, xmin=xmin, xmax=xmax)
        
    def vline(self, x, color='black', linewidth=1.0):
        ymin, ymax = self.ax.get_ylim()
        self.ax.vlines(x=x, color=color, linewidth=linewidth, ymin=ymin, ymax=ymax)
    
    def drawBar(self, time, value, colors=['green', 'red'], ylim=None, label=''):
        t0 = awarePyTime2Float(time[0])
        t1 = awarePyTime2Float(time[9])
        w = (t1 - t0) * 0.9 / 10.0
        for t, v in zip(time, value):
            if v is not None:
                if v >= 0:
                    color = colors[0]
                else:
                    color = colors[1]
                obj = BoxGraphic(t, w, v, color)
                obj.setObject(self.ax)
        if ylim is not None:
            self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.grid()
        return
    
    def drawBar2(self, time, value, color='red', ylim=None):
        t = []
        v = []
        for tt, vv in zip(time, value):
            t.append(tt)
            if vv is None:
                v.append(0)
            else:
                v.append(vv)
                
        t0 = awarePyTime2Float(t[0])
        t1 = awarePyTime2Float(t[1])
        self.ax.bar(t, v, color=color, width= (t1 - t0) * 0.9)
        if ylim is not None:
            self.ax.set_ylim(ylim[0], ylim[1])
        self.ax.grid()
        return
    
    def drawMarker(self, time, value, marker, color, markersize=10):
        t = awarePyTime2Float(time)
        self.ax.plot(t, value, marker=marker, color=color, markersize=markersize)
        return
    
    def xlimit(self, trange):
        x0 = awarePyTime2Float(trange[0])
        x1 = awarePyTime2Float(trange[1])
        self.ax.set_xlim(x0, x1)
        
    def ylimit(self, yrange):
        self.ax.set_ylim(yrange[0], yrange[1])
        
class BandPlot:
    def __init__(self, fig, ax, title, date_format=DATE_FORMAT_TIME):
        self.fix = fig
        self.ax = ax
        self.title = title
        self.ax.grid()
        self.ax.xaxis_date()
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        pass
        
    def boxWidth(self, time1, time2):
        t1 = awarePyTime2Float(time1)
        t2 = awarePyTime2Float(time2)
        return t2 - t1
    
    def drawBand(self, time, status, colors=None):
        if colors is None:
            colors = ['black', 'blue', 'red', 'pink', 'green', 'cyan', 'brown']
        self.ax.set_title(self.title)
        n = len(time)
        if n < 2:
            return
        
        box_width = self.boxWidth(time[0], time[1])
        self.graphic_objects = []
        for i in range(n):
            t = time[i]
            s = status[i]
            if s > 0:
                c = colors[s]
                obj = BoxGraphic(t, box_width, c)
                obj.setObject(self.ax)
                self.graphic_objects.append(obj)  
        self.ax.autoscale_view()
        return        
        
    def drawLine(self, time, value, color='red', linestyle='solid', linewidth=1.0):
        self.ax.plot(time, value, color=color, linestyle=linestyle, linewidth=linewidth)
        return     
    
    def xlimit(self, trange):
        x0 = awarePyTime2Float(trange[0])
        x1 = awarePyTime2Float(trange[1])
        self.ax.set_xlim(x0, x1)
        
#-----------
    
def test():
    
    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    plot = CandlePlot(fig, ax, 'test')
    
    t = [datetime(2021, 7, 1, 12, 0, 0),
         datetime(2021, 7, 1, 12, 5, 0)]
    ohlc = [[30050, 30300, 30000, 30100],
            [29900, 29950, 29800, 29850]]
    plot.drawCandle(t, ohlc, bar_width = 0.00002)

    plot.drawLine(t, [28000, 28500], color='black', linewidth=2.0)
    
    
    
    
    
if __name__ == '__main__':
    test()


