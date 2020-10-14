# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:24:55 2019

@author: wwj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings("ignore")

PATH='.'
onlyfiles = [f for f in listdir(PATH) if isfile(join(PATH, f))]
onlyfiles= [s.replace('dsp201607' , '') for s in onlyfiles]

for d in range(1):
    date=onlyfiles[1]
    #讀檔
    data_DSP_=pd.read_table(date,names=[0])
    print("1")
#    data_DSP_=data_DSP_.iloc[:100]
    #做一個panda的data_frame
    data_DSP=pd.DataFrame()
    def columns_read(name,j,k):
        data_DSP[name]=data_DSP_[0].str[j:k]

    cols_name=['證券代號','揭示時間','揭示註記','趨勢註記','成交揭示','成交漲跌停註記','成交價格','成交張數','檔位數','漲跌停註記','5檔買進價格及張數','5檔賣出價格及張數','揭示日期','撮合人員']
    cols_num=[6, 8, 1,1,1,1,6,8,1,1,70,70,8,2]
    
    flag=0
    for i in range(len(cols_name)):
        print(i)
        columns_read(cols_name[i],flag,flag+cols_num[i])
        flag+=cols_num[i]
    
    del cols_name,cols_num  