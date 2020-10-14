# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 20:38:33 2019

@author: wwj
"""

import pandas as pd
import numpy as np
import glob
import datetime

#%%
#load df
df_16_1 = pd.read_pickle("./Desktop/0826/year_dt/2016-1_dt.pkl")
df_16_2 = pd.read_pickle("./Desktop/0826/year_dt/2016-2_dt.pkl")
df_17_1 = pd.read_pickle("./Desktop/0826/year_dt/2017-1_dt.pkl")
df_17_2 = pd.read_pickle("./Desktop/0826/year_dt/2017-2_dt.pkl")
df_18_1 = pd.read_pickle("./Desktop/0826/year_dt/2018-1_dt.pkl")

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
    return final

#%%
close = LOAD_F()


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
def CLOSE_PRICE(close,get_df,year,month,day,hour,minute,second):
    date = datetime.datetime(year,month,day,hour,minute,second)
    
    close['DT'] = close['Date']+close['Time']
    close['DT'] = pd.to_datetime(close['DT'], format="%Y-%m-%d%H:%M:%S")
    
    am = datetime.datetime(year,month,day,8,45,second)
    pm = datetime.datetime(year,month,day,13,44,second)
    
    if date < am:
        #get yesterday 13:44 close price
        pre_date = datetime.datetime(year,month,day,13,44,second)
        pre_close = 
    elif date > pm:
        #get next day 8:45 close price
        askdjasd
    else:
        #get certain time
        idx = close.index[close['DT'] == date]
    




#%%
def SPOINT(df,year,month,day,hour,minute,second):
    #the time point we ask for
    date = datetime.datetime(year,month,day,hour,minute,second)
    get_df = df.loc[df['DT'] == str(date)]
    get_df = CLOSE_PRICE(close,get_df,date)
    
    return get_df
    
#%%
def PERIOD(df,y1,m1,d1,hour1,min1,sec1,y2,m2,d2,hour2,min2,sec2):
    date1 = datetime.datetime(y1,m1,d1,hour1,min1,sec1)
    date2 = datetime.datetime(y2,m2,d2,hour2,min2,sec2)
    if date1 > date2:
        print("invalid range of time!!!")
        get_df = pd.DataFrame()
    else:
        df['d1'] = str(date1)
        df['d2'] = str(date2)
        get_df = df.loc[df['DT'] >= df['d1']]
        get_df = get_df.loc[get_df['DT'] <= get_df['d2']]
        del get_df['d1']
        del get_df['d2']
    
    #output result to .csv
    get_df.to_csv("result.csv")
    
    return get_df
    
#%%
import datetime
yy = SPOINT(df_1701,2017,1,18,13,45,0)
zz = PERIOD(df_1801,2018,1,17,11,30,0,2018,1,18,11,31,0)











