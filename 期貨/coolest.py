# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 01:47:50 2019

@author: wwj
"""

import pandas as pd
import numpy as np
#%%
#import random
from multiprocessing import  Pool
def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 4)
    pool = Pool(4)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
#%%
def TXF(df):
    #df example: (df name)
    #get TXF only
    w = df['OSF_PROD_ID'].str.startswith("TXF")
    w = pd.DataFrame(w)
    w.columns = ['TXF']
    w = pd.concat([df,w],axis = 1, ignore_index=False)      
    df = w[w.TXF == True]  
    del df['TXF']
    return df

#%%
def REMAIN(temp,time):
    #time example: 201601 got from FWOSF
    #temp example: (df name)
    #calculate time remains
    year = int(str(time)[2:4])
    #solving index
    if year == 16:
        swidx = 0
    else:
        swidx = int((year-16)*12)
    #exp day this month
    exp1 = exp_ALL[swidx + int(str(time)[4:6])-1]+'13:30'
    #exp day next month
    exp2 = exp_ALL[swidx + int(str(time)[4:6])]+'13:30'
    #change format to datetime
    exp1 = pd.to_datetime(exp1,format = "%Y%m%d%H:%M")
    exp2 = pd.to_datetime(exp2,format = "%Y%m%d%H:%M")
    
    #split OSF_ORIG_TIME and get ['time']
    temp = TIME(temp)
    #'day' astype string + 'time'
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #hm col: h:m
    #temp['DT'] = temp['OSF_DATE'].astype(str)+temp['hm']
    #hms col: h:m:s
    temp['DT'] = temp['OSF_DATE'].astype(str)+temp['hms']
    
    #change format to datetime
    #for 'hm'
    #temp['DT'] = pd.to_datetime(temp['DT'],format = "%Y%m%d%H:%M")
    #for 'hms'
    temp['DT'] = pd.to_datetime(temp['DT'],format = "%Y%m%d%H:%M:%S")
    #end of!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #ergh
    temp['remain'] = 0
    temp['f1'] = exp1 - temp['DT']
    temp['f2'] = exp2 - temp['DT']
    
    temp['remain'] = np.where(temp['DT'] <= exp1, temp['f1'], temp['remain'])
    temp['remain'] = np.where(temp['DT'] > exp1, temp['f2'], temp['remain'])
    
    #calculate 'remain' using days+(seconds to days)
    temp['remain'] = temp['remain'].dt.days + temp['remain'].dt.seconds/(3600*24)
    return temp

#%%
def DATE_DATA_PROCESS(time,temp):
    #time example: 201601 got from FWOSF
    #temp example: (df name)
    temp = TXF(temp)
    
    #sort by time
    temp = temp.sort_values(by='OSF_ORIG_TIME', ascending=True)
    #reset index
    temp = temp.reset_index()
    del temp['index']
    
    temp = REMAIN(temp,time)
    #get a df of one date
    return temp
    
#%%
import glob

def FWOSF(time):
    #time example: 201601
    #dfname = 'df_'+str(time)
    
    #path for all FWOSF files and directories
    path = 'D:\chrome_download\FWOSF_'
    path = path+str(time)+'\*.txt'
    
    #all filenames list
    allfiles = glob.glob(path)
    
    #initial a df
    dfname = pd.DataFrame(columns=['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O'])  
    
    #read each day file
#    for i in range(10,12):
    for i in range(len(allfiles)):
        print(allfiles[i],i)
        #read date from a file name
        
        #read txt file and split by \t
        temp = pd.read_csv(allfiles[i], sep='\t', header=None)
        #name the columns
        temp.columns=['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O']
        
        #HERE WE GOT A DATAFRAME: temp
        #=============================function=================================
        #-----------------------------TXF(temp)--------------------------------
        #get TXF only
#        w = temp['OSF_PROD_ID'].str.startswith("TXF")
#        w = pd.DataFrame(w)
#        w.columns = ['TXF']
#        w = pd.concat([temp,w],axis = 1, ignore_index=False)      
#        temp = w[w.TXF == True]  
#        del temp['TXF']
        #=============================function=================================
        
#        #sort by time
#        temp = temp.sort_values(by='OSF_ORIG_TIME', ascending=True)
#        
#        #reset index
#        temp = temp.reset_index()
#        del temp['index']
        
        #=============================function=================================
        #------------------------REMAIN(temp,time)-----------------------------
        #calculate time remains
#        year = int(str(time)[2:4])
#        #solving index
#        if year == 16:
#            swidx = 0
#        else:
#            swidx = int((year-16)*12)
#        #exp day this month
#        exp1 = exp_ALL[swidx + int(str(time)[4:6])-1]+'13:30'
#        #exp day next month
#        exp2 = exp_ALL[swidx + int(str(time)[4:6])]+'13:30'
#        #change format to datetime
#        exp1 = pd.to_datetime(exp1,format = "%Y%m%d%H:%M")
#        exp2 = pd.to_datetime(exp2,format = "%Y%m%d%H:%M")
#        
#        #split OSF_ORIG_TIME and get ['time']
#        temp = TIME(temp)
#        #'day' astype string + 'time'
#        temp['DT'] = temp['OSF_DATE'].astype(str)+temp['time']
#        #change format to datetime
#        temp['DT'] = pd.to_datetime(temp['DT'],format = "%Y%m%d%H:%M")
#
#        #ergh
#        temp['remain'] = 0
#        temp['f1'] = exp1 - temp['DT']
#        temp['f2'] = exp2 - temp['DT']
#        
#        temp['remain'] = np.where(temp['DT'] <= exp1, temp['f1'], temp['remain'])
#        temp['remain'] = np.where(temp['DT'] > exp1, temp['f2'], temp['remain'])
#        
#        #calculate 'remain' using days+(seconds to days)
#        temp['remain'] = temp['remain'].dt.days + temp['remain'].dt.seconds/(3600*24)
        #=============================function=================================
        
        #df puts in parallelize_dataframe(df, func) function
        
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #CHECK TIME
        #  %%timeit
        FIN = DATE_DATA_PROCESS(time,temp) 
        #or
        #  %%timeit
        #FIN = DATE_DATA_PROCESS(temp)
        dfname = pd.concat([dfname,FIN], axis=0, ignore_index=False)
        #++++++++++++++++++++++++choose one(up or down)++++++++++++++++++++++++
        #using 1 function to process one day data
        temp = DATE_DATA_PROCESS(time,temp)
        
        #concat one day df to a month df
        dfname = pd.concat([dfname,temp], axis=0, ignore_index=False)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
                
    #delete columns
    del dfname['FCM_NO+ORDER_NO+O']
    del dfname['OSF_DEL_QNTY']
    del dfname['OSF_UN_QNTY']
    del dfname['OSF_ORDER_PRICE']
    del dfname['OSF_ORDER_TYPE']
    del dfname['OSF_OC_CODE']
    del dfname['OSF_SEQ_NO']
    del dfname['OSF_PROD_ID']
    
    del dfname['OSF_DATE']
    del dfname['OSF_ORIG_TIME']
    
    return dfname

#%%
def TIME(df):
    #df example: df_1601    
    #split 'OSF_ORIG_TIME' using '.'
    t = df['OSF_ORIG_TIME'].str.split(".", n = 1, expand = True)
    #name columns
    t.columns = ['h-m-s','other']
    
    
    #split 'h-m-s' using ':' twice
    hms = t['h-m-s'].str.split(":", n = 2, expand = True)
    
    del t['h-m-s']
    #t col: other
    
    #name columns
    hms.columns = ['h','m','s']
    
    #new col 'time' = 'h':'m'
    #t['time'] = t['h']+':'+t['m']
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    hms['hm'] = hms['h'] + ':' + hms['m']
    hms['hms'] = hms['h'] + ':' + hms['m'] + ':' + hms['s']
    #hms col: h,m,s,hm,hms
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    del hms['h']
    del hms['m']
    del hms['s']
    #hms col: hm,hms
    
    #delete columns
    #del t['h']
    #del t['m']
    #del t['s']
    #concat back to original df
    #df = pd.concat([df,t], axis=1, ignore_index=False)
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    df = pd.concat([df,hms], axis=1, ignore_index=False)
    
    #delete 'OSF_ORIG_TIME'
    del df['OSF_ORIG_TIME']
    
    return df

#%%
def BS(df):
    #df example: df_1601
    #set Buy as 0
    df.loc[df['OSF_BS_CODE'] == 'B', 'OSF_BS_CODE'] = 0
    #set Sell as 1
    df.loc[df['OSF_BS_CODE'] == 'S', 'OSF_BS_CODE'] = 1
    return df

#%%
def OH(df):
    #df example: df_1601
    #ROD IOC FOK
    #one hot for 'OSF_ORDER_COND'
    ORDER_COND = pd.get_dummies(df['OSF_ORDER_COND'])
    #concat back to original df
    df = pd.concat([df,ORDER_COND], axis=1, ignore_index=False)
    #delete column 'OSF_ORDER_COND'
    del df['OSF_ORDER_COND']
    return df

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
def FLOAT(df):
    #df example: df_1601
    #change 'remain' representation as .2f
    df['remain'] = df['remain'].astype(float).map('{:,.2f}'.format)
    return df

#%%
def to_do(time):
    #example:201601    
    #define a dataframe
    df = []
    #time as string type
    time = str(time)
    #call the functions
    df = FWOSF(time)
    df = BS(df)
    df = OH(df)
    #change the 'remain' column representation
    df = FLOAT(df)
    #save the result
    df.to_csv(time+'05.csv')
    
    ALL = pd.DataFrame() 
    ALL = pd.concat([ALL,df], axis=0, ignore_index=False)
    
    return ALL,df

#%%    
ALL,df_1601 = to_do(201601)
ALL,df_1602 = to_do(201602)
ALL,df_1603 = to_do(201603)
ALL,df_1604 = to_do(201604)
ALL,df_1605 = to_do(201605)
ALL,df_1606 = to_do(201606)
ALL,df_1607 = to_do(201607)
ALL,df_1608 = to_do(201608)
ALL,df_1609 = to_do(201609)
ALL,df_1610 = to_do(201610)
ALL,df_1611 = to_do(201611)
ALL,df_1612 = to_do(201612)

ALL_17,df_1701 = to_do(201701)
ALL,df_1702 = to_do(201702)
ALL,df_1703 = to_do(201703)
ALL,df_1704 = to_do(201704)
ALL,df_1705 = to_do(201705)
ALL,df_1706 = to_do(201706)
ALL,df_1707 = to_do(201707)
ALL,df_1708 = to_do(201708)
ALL,df_1709 = to_do(201709)
ALL,df_1710 = to_do(201710)
ALL,df_1711 = to_do(201711)
ALL,df_1712 = to_do(201712)

ALL,df_1801 = to_do(201801)
ALL,df_1802 = to_do(201802)

#%%
#testing
test = to_do(201701)
#%%

#concat all
ALL_2016 = pd.concat([df_1601,df_1602.df_1603,
                 df_1604,df_1605,df_1606,
                 df_1607,df_1608,df_1609,
                 df_1610,df_1611,df_1612], axis=0, ignore_index=False)
ALL_2017 = pd.concat([df_1701,df_1702,df_1703,
                 df_1704,df_1705,df_1706,
                 df_1707,df_1708,df_1709,
                 df_1710,df_1711,df_1712], axis=0, ignore_index=False)
ALL_2018 = pd.concat([df_1801,df_1802], axis=0, ignore_index=False)

ALL = pd.concat([ALL_2016,ALL_2017,ALL_2018], axis=0, ignore_index=False)

ALL.to_csv("all.csv")

#%%
#label y: B, S, OSF_ORDER_QNTY

#%%
import datetime
def SPOINT(df,year,month,day,hour,minute,second):
    if year <16 or year>18:
        print("year is invalid!!!")
        get_df = pd.DataFrame()
    else:
        #the time point we ask for
        date = datetime.datetime(year,month,day,hour,minute,second)
        #calculate days before next exp day
        month = int(month)
        #solving index
        if year == 2016:
            swidx = 0
        else:
            swidx = int(((year%2000)-16)*12)
        exp1 = exp_ALL[swidx+month-1]+'13:30'
        exp2 = exp_ALL[swidx+month]+'13:30'
        exp1 = pd.to_datetime(exp1,format = "%Y%m%d%H:%M")
        exp2 = pd.to_datetime(exp2,format = "%Y%m%d%H:%M")
        if date <= exp1:
            d = exp1 - date
        elif date > exp1:
            d = exp2 - date
        d = d.days+d.seconds/(3600*24)
        #d = d.map('{:,.2f}'.format)
        dd = round(d, 2)
        dd = str(dd)
        get_df = df.loc[df['remain'] == dd ]
        
    return get_df
    
#%%
def PERIOD(df,y1,m1,d1,hour1,min1,sec1,y2,m2,d2,hour2,min2,sec2):
    date1 = datetime.datetime(y1,m1,d1,hour1,min1,sec1)
    date2 = datetime.datetime(y2,m2,d2,hour2,min2,sec2)
    if date1 > date2:
        print("invalid range of time!!!")
        get_df = pd.DataFrame()
    else:
        df['d1'] = date1
        df['d2'] = date2
        get_df = df.loc[df['DT'] >= df['d1']]
        get_df = get_df.loc[get_df['DT'] <= get_df['d2']]
        del get_df['d1']
        del get_df['d2']
    
    #output result to .csv
    get_df.to_csv("result.csv")
    
    return get_df
    
#%%
yy = SPOINT(df_1701,2017,1,18,13,45,0)
zz = PERIOD(df_1701,2017,1,17,11,30,0,2017,1,18,11,31,0)



