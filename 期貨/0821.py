# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:44:38 2019

@author: wwj
"""

#%%
import pandas as pd
import numpy as np
import datetime

#%%
#def DAY(df):
#    #df example: df_1601
#    df['days']=0
#    for i in range(len(df)):
#        df.iloc[i]['days'] = d1601.day - int(df.iloc[i]['d'])
#        df.loc[df['days'] < 0 , 'days'] = d1602.day - int(df.iloc[i]['d'])
#    return df
#%%
import glob
#allfiles = glob.glob('.\*.txt')
#df_1601 = pd.DataFrame(columns=['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O'])
def FWOSF(time):
    #time example: 201601
    #dfname = 'df_'+str(time)
    path = 'D:\chrome_download\FWOSF_'
    path = path+str(time)+'\*.txt'
    allfiles = glob.glob(path)
    dfname = pd.DataFrame(columns=['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O'])  
    for i in range(1):
    #for i in range(len(allfiles)):
        print(i)
        DATE = allfiles[i][-12:-4]
        
        #read file
        temp = pd.read_csv(allfiles[i], sep='\t', header=None)
        temp.columns=['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O']
        #get TXF
        w = temp['OSF_PROD_ID'].str.startswith("TXF")
        w = pd.DataFrame(w)
        w.columns = ['TXF']
        w = pd.concat([temp,w],axis = 1, ignore_index=False)      
        temp = w[w.TXF == True]  
        del temp['TXF']
        #sort by time
        temp = temp.sort_values(by='OSF_ORIG_TIME', ascending=True)
        #reset index
        temp = temp.reset_index()
        del temp['index']
        
        #time remains
        exp1 = d_all_str[int(str(time)[4:6])-1]+'13:30'
        exp2 = d_all_str[int(str(time)[4:6])]+'13:30'
        exp1 = pd.to_datetime(exp1,format = "%Y%m%d%H:%M")
        exp2 = pd.to_datetime(exp2,format = "%Y%m%d%H:%M")
        
        temp = TIME(temp)
        temp['DT'] = temp['OSF_DATE'].astype(str)+temp['time']
        temp['DT'] = pd.to_datetime(temp['DT'],format = "%Y%m%d%H:%M")
        
        if DATE < d_all_str[int(str(time)[4:6])-1]:
            temp['remain'] = exp1 - temp['DT']
        else:
            temp['remain'] = exp2 - temp['DT']
        
        temp['remain'] = temp['remain'].dt.days + temp['remain'].dt.seconds/(3600*24)
        #concat
        dfname = pd.concat([dfname,temp], axis=0, ignore_index=False)
        #print(dfname)        
    #delete columns
    del dfname['FCM_NO+ORDER_NO+O']
    del dfname['OSF_DEL_QNTY']
    del dfname['OSF_UN_QNTY']
    del dfname['OSF_ORDER_PRICE']
    del dfname['OSF_ORDER_TYPE']
    del dfname['OSF_OC_CODE']
    del dfname['OSF_SEQ_NO']
    del dfname['OSF_PROD_ID']
    
    del dfname['DT']
    del dfname['OSF_DATE']
    del dfname['OSF_ORIG_TIME']
    
    return dfname

#%%
def TIME(df):
    #df example: df_1601
    t = df['OSF_ORIG_TIME'].str.split(".", n = 1, expand = True)
    t.columns = ['h-m-s','other']
    t = t['h-m-s'].str.split(":", n = 2, expand = True)
    t.columns = ['h','m','s']
    t['time'] = t['h']+':'+t['m']
    del t['h']
    del t['m']
    del t['s']
    df = pd.concat([df,t], axis=1, ignore_index=False)
    del df['OSF_ORIG_TIME']
    return df
#%%
def BS(df):
    df.loc[df['OSF_BS_CODE'] == 'B', 'OSF_BS_CODE'] = 0
    df.loc[df['OSF_BS_CODE'] == 'S', 'OSF_BS_CODE'] = 1
    return df
#%%
def OH(df):
    #df example: df_1601
    #ROD IOC FOK
    ORDER_COND = pd.get_dummies(df['OSF_ORDER_COND'])
    df = pd.concat([df,ORDER_COND], axis=1, ignore_index=False)
    del df['OSF_ORDER_COND']
#    df.columns = [['OSF_BS_CODE', 'OSF_ORDER_QNTY', 'remain', 'time','FOK','IOC','ROD']]
    return df
#%%
from datetime import date
#d16 = [date(2016,1,20),date(2016,2,17),date(2016,3,16),date(2016,4,20),
#     date(2016,5,18),date(2016,6,15),date(2016,7,20),date(2016,8,17),
#     date(2016,9,21),date(2016,10,19),date(2016,11,16),date(2016,12,21)]
#
#d17 = [date(2017,1,18),date(2017,2,15),date(2017,3,15),date(2017,4,19),
#     date(2017,5,18),date(2017,6,21),date(2017,7,19),date(2017,8,16),
#     date(2017,9,20),date(2017,10,18),date(2017,11,15),date(2017,12,20)]
#d18 = [date(2018,1,17),date(2018,2,21)]

d_all = [date(2016,1,20),date(2016,2,17),date(2016,3,16),date(2016,4,20),
     date(2016,5,18),date(2016,6,15),date(2016,7,20),date(2016,8,17),
     date(2016,9,21),date(2016,10,19),date(2016,11,16),date(2016,12,21),
     date(2017,1,18),date(2017,2,15),date(2017,3,15),date(2017,4,19),
     date(2017,5,18),date(2017,6,21),date(2017,7,19),date(2017,8,16),
     date(2017,9,20),date(2017,10,18),date(2017,11,15),date(2017,12,20),
     date(2018,1,17),date(2018,2,21)]

d_all_str = ['20160120','20160217','20160316','20160420',
     '20160518','20160615','20160720','20160817',
     '20160921','20161019','20161116','20161221',
     '20170118','20170215','20170315','20160419',
     '20170518','20170621','20170719','20170816',
     '20170920','20171018','20171115','20171220',
     '20180117','20180221']

#%%
d_all = pd.DataFrame(d_all)
d_all['s'] = '13:30'
d_all.columns = ['d','s']
d_all['ds'] = d_all['d'].astype(str)+d_all['s']
d_all['ds'] = pd.to_datetime(d_all['ds'],format = "%Y-%m-%d%H:%M")
##2016
#d1601 = date(2016,1,20)
#d1602 = date(2016,2,17)
#d1603 = date(2016,3,16)
#d1604 = date(2016,4,20)
#d1605 = date(2016,5,18)
#d1606 = date(2016,6,15)
#d1607 = date(2016,7,20)
#d1608 = date(2016,8,17)
#d1609 = date(2016,9,21)
#d1610 = date(2016,10,19)
#d1611 = date(2016,11,16)
#d1612 = date(2016,12,21)
##2017
#d1701 = date(2017,1,18)
#d1702 = date(2017,2,15)
#d1703 = date(2017,3,15)
#d1704 = date(2017,4,19)
#d1705 = date(2017,5,18)
#d1706 = date(2017,6,21)
#d1707 = date(2017,7,19)
#d1708 = date(2017,8,16)
#d1709 = date(2017,9,20)
#d1710 = date(2017,10,18)
#d1711 = date(2017,11,15)
#d1712 = date(2017,12,20)
##2018
#d1801 = date(2018,1,17)
#d1802 = date(2018,2,21)
#%%
#from datetime import date

#d_all = [date(2016,1,20),date(2016,2,17),date(2016,3,16),date(2016,4,20),
#     date(2016,5,18),date(2016,6,15),date(2016,7,20),date(2016,8,17),
#     date(2016,9,21),date(2016,10,19),date(2016,11,16),date(2016,12,21),
#     date(2017,1,18),date(2017,2,15),date(2017,3,15),date(2017,4,19),
#     date(2017,5,18),date(2017,6,21),date(2017,7,19),date(2017,8,16),
#     date(2017,9,20),date(2017,10,18),date(2017,11,15),date(2017,12,20),
#     date(2018,1,17),date(2018,2,21)]
#
#d_all_str = ['20160120','20160217','20160316','20160420',
#     '20160518','20160615','20160720','20160817',
#     '20160921','20161019','20161116','20161221',
#     '20170118','20170215','20170315','20160419',
#     '20170518','20170621','20170719','20170816',
#     '20170920','20171018','20171115','20171220',
#     '20180117','20180221']

#%%
def DATE(df,time):
    #df example: df_1601
    #time example: 201601
    #dd = df['OSF_DATE'].astype(int)
#    tt = int(time)
#    df['y'] = df['OSF_DATE'].floordiv(10000) #format(tt//100, '04')
#    df['m'] = df['OSF_DATE'].astype(str).str[4:6] #format(tt%100, '02')
#    df['d'] = df['OSF_DATE'].astype(str).str[6:]
#    df['date'] = df['y'].astype(str) + '-' + df['m'].astype(str) + '-' + df['d'].astype(str)
    #delete column
#    del df['y']
#    del df['m']
#    del df['d']
    df['datetime'] = df['OSF_DATE'].astype(str)+df['time']
#    df['datetime'] = datetime.datetime.strptime(df['datetime'],"%Y%m%d%H:%M")
    df['datetime'] = pd.to_datetime(df['datetime'],format='%Y%m%d%H:%M')
#    del df['OSF_DATE']
    return df

#%%
df_1601 = FWOSF(201601)
#df_1601 = TIME(df_1601)
df_1601 = BS(df_1601)
df_1601 = OH(df_1601)
#del df_1601['time']
#%%

#df_1601 = DATE(df_1601,201601)

#%%
df_1601.to_csv("1601.csv")
#%%
df_1601 = pd.read_csv("1601.csv")
#%%

df_1601_remain = DAYS(df_1601,201601)

#%%
def DAYS(df,time):
    #time example: 201601
    year = str(time)[2:4]
    month = str(time)[4:6]
    #lname = 'd'+year
    k=0
    
    if int(year) == 17:
        k = 12 + int(month)
    elif int(year) == 18:
        k = 24 + int(month)
    else:
        k = 0 + int(month)
    print(year,month,k)
    
    d1 = d_all.iloc[k-1]['ds']
    d2 = d_all.iloc[k]['ds']
    
    df['remains'] = d1 - df['datetime'] 
    
    
#    df['f1'] = d_all[k-1]
#    df['f2'] = d_all[k]
    
    
#    f1_date = df['f1'].values.astype('datetime64[D]')
#    f2_date = df['f2'].values.astype('datetime64[D]')
#    datee = df['date'].values.astype('datetime64[D]')
#    print(f1_date,f2_date,datee)
#    
#    df['before'] = f1_date - datee
#    before = df['before'].astype(str).str.split(" ", n = 1, expand = True)
#    before.columns = ['days','-']
#    del before['-']
#    df = pd.concat([df,before], axis=1, ignore_index=False)
#    
#    df['after'] = f2_date - datee
#    after = df['after'].astype(str).str.split(" ", n = 1, expand = True)
#    after.columns = ['days2','-']
#    del after['-']
#    df = pd.concat([df,after], axis=1, ignore_index=False)
#    
#    t1 = df.loc[df['days'].astype(int) < 0 ]
#    t1['days'] = t1['days2']
#    
#    t2 = df.loc[df['days'].astype(int) >= 0 ]
#    
#    df = pd.concat([t1,t2], axis=0, ignore_index=False)
#    df.drop(['before', 'after','days2','y','m','d','date','f1','f2'], axis = 1) 
    
#    del['before']
#    del['after']
#    del[]
#    del[]
#    del[]
#    del[]
#    del[]
    d2 - df['datetime']
    df.loc[df['remains'].astype(str).str.startswith("-"), 'remains'] = d2 - df['datetime'] 
#    a = df['before'].astype(str).str.startswith("-")
#    df.loc[df['before'].astype(str).str.startswith("-"), 'days'] = df['after']
#    df.loc[df['day2'] == 0 , 'day2'] = df['day1']
    
    
    return df


#%%
#load data

data = pd.read_csv('FWOSF_20160104.txt', sep='\t', header=None)
data_original = pd.read_csv('FWOSF_20160104.txt', sep='\t', header=None)
#data.columns = ['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O']
#data_original.columns = ['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O']
data.columns = ['日期','商品代號','買賣別','委託量','未成交量','委託價','委託方式','ORDER_COND','開平倉碼','委託時間','委託序號','減量口數','交易與委託報價檔連結代碼']
data_original.columns = ['日期','商品代號','買賣別','委託量','未成交量','委託價','委託方式','ORDER_COND','開平倉碼','委託時間','委託序號','減量口數','交易與委託報價檔連結代碼']
#%%
#find TXF
w = data_original["商品代號"].str.startswith("TXF")
w = pd.DataFrame(w)
w.columns = ['TXF']
w = pd.concat([data_original,w],axis = 1, ignore_index=False)      
txf = w[w.TXF == True]  
del txf['TXF'] 
del txf['商品代號']
#%%
del txf['未成交量']
del txf['委託序號']
del txf['交易與委託報價檔連結代碼']
del txf['委託價']
del txf['開平倉碼']
del txf['減量口數']
del txf['委託方式']
#%%
time = txf['委託時間'].str.split(".", n = 1, expand = True) 
time.columns = ['h-m-s','other']

time = time['h-m-s'].str.split(":", n = 1, expand = True)
time.columns = ['h','m:s']
h = time['h']
h = pd.DataFrame(h)
h.columns = ['h']
time = time['m:s'].str.split(":", n = 1, expand = True)
time.columns = ['m','s']
time = pd.concat([h,time], axis=1, ignore_index=False)
time['h:m'] = time['h']+':'+time['m']
txf = pd.concat([txf,time], axis=1, ignore_index=False)
del txf['委託時間']

#%%
# 上下五檔
future = pd.read_pickle('future_2016-01-04.pickle')
close = future[['Time','Close']]
close['u1'] = 0
close['u2'] = 0
close['u3'] = 0
close['u4'] = 0
close['u5'] = 0
close['d1'] = 0
close['d2'] = 0
close['d3'] = 0
close['d4'] = 0
close['d5'] = 0

#for i in range(5):
#    i=i+1
#    close['u'+str(i)] = 0
#    close['d'+str(i)] = 0
#
#for i in range(len(close)):
#    for i in range(5):
#        i=i+1
#        close['u'+str(i)][i] = close['Close'][i]+i
#        close['d'+str(i)][i] = close['Close'][i]-i

for i in range(len(close)):
    close['u1'][i] = close['Close'][i]+1
    close['u2'][i] = close['Close'][i]+2
    close['u3'][i] = close['Close'][i]+3
    close['u4'][i] = close['Close'][i]+4
    close['u5'][i] = close['Close'][i]+5
    close['d1'][i] = close['Close'][i]-1
    close['d2'][i] = close['Close'][i]-2
    close['d3'][i] = close['Close'][i]-3
    close['d4'][i] = close['Close'][i]-4
    close['d5'][i] = close['Close'][i]-5
    
#%%
#unfinished
def timetohm(df, field, step):
    temp = df[field].str.split(":", n = step, expand = True)
    if step == 2:
        temp.columns = ['h','m','s']
        temp['h:m'] = temp['h']+':'+temp['m']
    elif step == 1:
        temp.columns = ['time','other']
        temp = timetohm(temp, 'time',1)
    df = pd.concat([temp,df], axis=1, ignore_index=False)
    return df
#%%
close_time = close['Time'].str.split(":", n = 2, expand = True) 
close_time.columns = ['h','m','s']
close_time['h:m'] = close_time['h']+':'+close_time['m']
close = pd.concat([close,close_time], axis=1, ignore_index=False)
#%%
del close['h']
del close['m']
del close['s']
del close['Time']

#%%
del txf['h']
del txf['m']
del txf['s']
#%%
txf = txf.reset_index()
del txf['index']
#%%
#把市價單的委託價(市價)填入
#txf.loc [ txf ['委託價'] == 0  , '委託價'] = 0
#for i in range(len(txf)):
#    if txf.iloc[i]['委託價'] == txf.iloc[i]['h:m']:
#        for j in range(len(close)):
#            print(i,j)
#            t = txf.iloc[i]['h:m']
#            d = close[close['h:m'] == t]
#            txf.iloc[i]['委託價'] = d['Close']

#for i in range(len(txf)):
#    if txf.iloc[i]['委託價'] == 0:
#        t = txf.iloc[i]['h:m']
#        d = close[close['h:m'] == t]
#        a = d.iloc[0]['Close']
#        txf.iloc[i]['委託價'] = a

#%%
#部分欄位做one-hot
#ROD IOC FOK Q U
ORDER_COND = pd.get_dummies(txf['ORDER_COND'])
txf = pd.concat([txf,ORDER_COND], axis=1, ignore_index=False)
del txf['ORDER_COND']
#%%
#買賣
BS_CODE = pd.get_dummies(txf['買賣別'])

#%%
#ignore index false, 左右合併
def concat(df1,df2):
#    cc = pd.concat([df1,df2], axis=1, ignore_index=False)
#    return cc
#%%
#計算距離到期日有多遠(每個txt檔日期一樣)
from datetime import date
#2016
d1601 = date(2016,1,20)
d1602 = date(2016,2,17)
d1603 = date(2016,3,16)
d1604 = date(2016,4,20)
d1605 = date(2016,5,18)
d1606 = date(2016,6,15)
d1607 = date(2016,7,20)
d1608 = date(2016,8,17)
d1609 = date(2016,9,21)
d1610 = date(2016,10,19)
d1611 = date(2016,11,16)
d1612 = date(2016,12,21)
#2017
d1701 = date(2017,1,18)
d1702 = date(2017,2,15)
d1703 = date(2017,3,15)
d1704 = date(2017,4,19)
d1705 = date(2017,5,18)
d1706 = date(2017,6,21)
d1707 = date(2017,7,19)
d1708 = date(2017,8,16)
d1709 = date(2017,9,20)
d1710 = date(2017,10,18)
d1711 = date(2017,11,15)
d1712 = date(2017,12,20)
#2018
d1801 = date(2018,1,17)
d1802 = date(2018,2,21)

#delta = d1 - d0
#print (delta.days)

#%%
da = future.iloc[0]['Date']
da = datetime.strptime(da, '%Y-%m-%d').date()
txf['days'] = (d1601-da).days
del txf['日期']
#%%
#change column order
txf = txf[['days','h:m','買賣別','F','I','R','委託量']]
txf.columns = ['days','time','買賣別','FOK','IOC','ROD','委託量']
#%%
txf = txf.sort_values(by='h:m', ascending=True)
txf = txf.reset_index()
del txf['index']

#%%
txf.loc[txf['買賣別'] == 'B', '買賣別'] = 0
txf.loc[txf['買賣別'] == 'S', '買賣別'] = 1

#%%















