# -*- coding: utf-8 -*-

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../common'))
import datetime



def holderList(path):
    holder= []
    for filename in os.listdir(path):
        if os.path.isdir(os.path.join(path, filename)):
            holder.append(filename)
    return holder


def isHolder(market, year):
    holder_list = holderList('../click_sec_data/' + market)
    
    #tend = datetime.now()
    
    for month in range(1, 13):
        t = datetime.datetime(year, month, 1)
        now = datetime.datetime.now()
        t1 = datetime.datetime(now.year, now.month, 1)
        if t > t1:
            break
        
        name = market+ '_' + str(year) + str(month).zfill(2)
        if not name in holder_list:
            print('No holder:', name)
        

def fileCheck():
    
    markets = ['CHNA50', 'EURO_50_Index', 'GER40', 'HK', 'JP225', 'SPOT_GOLD', 'UK100', 'US30', 'US500', 'USTEC', 'vix', 'WTI']
    
    for market in markets:
        for year in [2019, 2020, 2021]:
            isHolder(market, year)
    
    
    
if __name__ == '__main__':
    fileCheck()
    
    

