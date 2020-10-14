# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 15:38:57 2019

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
    data_DSP_=pd.read_table('./'+date)
#    data_DSP_=data_DSP_.iloc[:100]
    #做一個panda的data_frame
    data_DSP=pd.DataFrame()
    def columns_read(name,j,k):
        data_DSP[name]=data_DSP_[0].str[j:k]

    cols_name=['股票代號','揭示時間','-','Bid1','Bidlots1','Bid2','Bidlots2','Bid3','Bidlots3','Bid4','Bidlots4','Bid5','Bidlots5','--','Ask1','Asklots1','Ask2','Asklots2','Ask3','Asklots3','Ask4','Asklots4','Ask5','Asklots5','日期','---']
    cols_num=[6, 8, 20, 6, 8, 6, 8,6, 8,6, 8,6, 8,2,6, 8,6, 8,6, 8,6, 8,6, 8,8,2]
    
    flag=0
    for i in range(len(cols_name)):
        print(i)
        columns_read(cols_name[i],flag,flag+cols_num[i])
        flag+=cols_num[i]
    
    del cols_name,cols_num  