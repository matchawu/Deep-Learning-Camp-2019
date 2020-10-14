# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 16:31:48 2019

@author: wwj
"""

import pandas as pd
import numpy as np
import glob

#%%
import calendar

def WED(year):
    # year example: 2016
    #initial exp day list
    exp_list = []
    #find third WED each month in a year
    for month in range(1, 13):
        cal = calendar.monthcalendar(year, month)
        first_week  = cal[0]
        #second_week = cal[1]
        third_week  = cal[2]
        forth_week  = cal[3]       
        # If a Wednesday presents in the first week, the third Wednesday
        # is in the third week.  Otherwise, the third Wednesday must 
        # be in the forth week.        
        if first_week[calendar.WEDNESDAY]:
            exp_day = third_week[calendar.WEDNESDAY]
        else:
            exp_day = forth_week[calendar.WEDNESDAY]    
        #print('%2s/%2s' % (str(month).zfill(2), exp_day))        
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
def READ(time):
    #time example: 20160104
df = pd.read_pickle("future_2016-01-04.pickle")
date = df['Date']
time = df['Time']
df = pd.concat([date,time], axis=1, ignore_index=False)

#%%
def TIME(df):
    #df example: df_1601    
    #split 'OSF_ORIG_TIME' using '.'
    t = df['OSF_ORIG_TIME'].str.split(".", n = 1, expand = True)
    #name columns
    t.columns = ['h-m-s','other']
    #split 'h-m-s' using ':' twice
    t = t['h-m-s'].str.split(":", n = 2, expand = True)
    #name columns
    t.columns = ['h','m','s']
    #new col 'time' = 'h':'m'
    t['time'] = t['h']+':'+t['m']
    #delete columns
    del t['h']
    del t['m']
    del t['s']
    #concat back to original df
    df = pd.concat([df,t], axis=1, ignore_index=False)
    #delete 'OSF_ORIG_TIME'
    del df['OSF_ORIG_TIME']
    
    return df

#%%
def REMAIN(time,temp):
    #calculate time remains (minutes)
    year = int(str(time)[2:4])
    #solving index
    swidx = int((year-16)*12)
    #exp day this month
    exp1 = exp_ALL[swidx + int(str(time)[4:6])-1]+'13:30'
    #exp day next month
    exp2 = exp_ALL[swidx + int(str(time)[4:6])]+'13:30'
    #change format to datetime
    exp1 = pd.to_datetime(exp1,format = "%Y%m%d%H:%M")
    exp2 = pd.to_datetime(exp2,format = "%Y%m%d%H:%M")
    
    #split OSF_ORIG_TIME and get 'time'
    temp = TIME(temp)
    #'day' astype string + 'time'
    temp['DT'] = temp['OSF_DATE'].astype(str)+temp['time']
    #change format to datetime
    temp['DT'] = pd.to_datetime(temp['DT'],format = "%Y%m%d%H:%M")

    #ergh
    temp['remain'] = 0
    temp['f1'] = exp1 - temp['DT']
    temp['f2'] = exp2 - temp['DT']
    
    temp['remain'] = np.where(temp['DT'] <= exp1, temp['f1'], temp['remain'])
    temp['remain'] = np.where(temp['DT'] > exp1, temp['f2'], temp['remain'])
    
    #calculate 'remain' using days+(seconds to days)
    temp['remain'] = temp['remain'].dt.days + temp['remain'].dt.seconds/(3600*24)
    
    return temp