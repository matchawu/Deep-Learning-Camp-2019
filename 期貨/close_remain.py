# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 22:05:17 2019

@author: wwj
"""
import pandas as pd
import numpy as np
import glob
import datetime
import calendar
#%%
pp = pd.read_pickle("./Desktop/0826/nm_futures_minutes/future_2016-11-09.pickle")
qq = pd.read_pickle("./Desktop/0826/nm_futures_minutes/future_2018-02-06.pickle")
#%%
#load futures
def LOAD_F():
    #path for all FWOSF files and directories
    path = './Desktop/0826/nm_futures_minutes'
    path = path+'/*.pickle'
    allfiles = glob.glob(path)
    final = pd.DataFrame(columns=['Date','Time','Close'])
    for i in range(len(allfiles)):
        temp = pd.read_pickle(allfiles[i])
        temp = temp[['Date','Time','Close']]
        final = pd.concat([final,temp], axis=0, ignore_index=False)
        final = final.reset_index(drop=True)
    return final

#%%
close = LOAD_F()
#%%
c = close
c = c.drop([61800,153301],axis=0)
c = c.reset_index(drop=True)
#%%
def WED(year):
    # year example: 2016
    #initial exp day list
    exp_list = []
    for month in range(1, 13):
        cal = calendar.monthcalendar(year, month)
        first_week  = cal[0]
        third_week  = cal[2]
        forth_week  = cal[3]    
        if first_week[calendar.WEDNESDAY]:
            exp_day = third_week[calendar.WEDNESDAY]
        else:
            exp_day = forth_week[calendar.WEDNESDAY]     
        exp_date = str(year)+str(month).zfill(2)+str(exp_day)
        exp_list.append(exp_date)
        
    return exp_list
    
#%%
#get all third WED in each year
exp_2016 = WED(2016)
exp_2017 = WED(2017)
exp_2018 = WED(2018)

#initial list for all exp days
#201601 - 201802
exp_ALL = []
#aware of "append" and "expand" are not the smae
exp_ALL.extend(exp_2016)
exp_ALL.extend(exp_2017)
exp_ALL.extend(exp_2018)
#%%
def REMAIN(df):
    df['DT'] = df['Date']+" "+df['Time']
    df['DT'] = pd.to_datetime(df['DT'], format="%Y-%m-%d %H:%M:%S")
    df['REMAIN'] = 0
    for i in range(len(df)):
        print(i)
        date = df.iloc[i]['DT']
        year = date.year
        month = date.month
        swidx = int((year%2000-16)*12)+month
        exp1 = exp_ALL[swidx-1]+'13:30'
        exp2 = exp_ALL[swidx]+'13:30'
        exp1 = pd.to_datetime(exp1,format = "%Y%m%d%H:%M")
        exp2 = pd.to_datetime(exp2,format = "%Y%m%d%H:%M")
        if date > exp1:
            remain = exp2 - date
        elif date <= exp1:
            remain = exp1 - date
        df.loc[i,'REMAIN'] = remain.days + remain.seconds/(3600*24)
        
    df['REMAIN'] = df['REMAIN'].astype(float).map('{:,.2f}'.format)
    return df

#%%
c = REMAIN(close)
        

