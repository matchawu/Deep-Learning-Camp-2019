# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 18:26:24 2019

@author: wwj
"""
#%%
import pandas as pd
import numpy as np

#%%
data = pd.read_csv('FWOSF_20160104.txt', sep='\t', header=None)
data_original = pd.read_csv('FWOSF_20160104.txt', sep='\t', header=None)
data.columns = ['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O']


data_BS_CODE = pd.get_dummies(data['OSF_BS_CODE'])
data_ORDER_TYPE = pd.get_dummies(data['OSF_ORDER_TYPE'])
data_ORDER_COND = pd.get_dummies(data['OSF_ORDER_COND'])
data_OC_CODE = pd.get_dummies(data['OSF_OC_CODE'])

data.columns = ['日期','商品代號','買賣別','委託量','未成交量','委託價','委託方式','ORDER_COND','開平倉碼','委託時間','委託序號','減量口數','交易與委託報價檔連結代碼']
data_original.columns = ['日期','商品代號','買賣別','委託量','未成交量','委託價','委託方式','ORDER_COND','開平倉碼','委託時間','委託序號','減量口數','交易與委託報價檔連結代碼']
del data['委託序號']
del data['交易與委託報價檔連結代碼']

del data['買賣別']
del data['委託方式']
del data['ORDER_COND']
del data['開平倉碼']

data = pd.concat([data,data_BS_CODE,data_ORDER_TYPE,data_ORDER_COND,data_OC_CODE], axis=1, ignore_index=False)
data.columns = ['日期','商品代號','委託量','未成交量','委託價','委託時間','減量口數','買','賣','MWP','限價單','市價單','FOK','IOC','Q','ROD','U','新倉','平倉',2,9]
#data_TXF = data.loc[data['商品代號'].str.startswith('TXF'), "Datum"].values[0]
#%%
time = data['委託時間'].str.split(".", n = 1, expand = True) 
time.columns = ['h-m-s','other']

time = time['h-m-s'].str.split(":", n = 1, expand = True)
time.columns = ['h','m:s']
data_time = pd.concat([data,time], axis=1, ignore_index=False)
del data_time['委託時間']
del data_time['m:s']
df_7 = data_time

#%%
# hour
df_7 = data_time[data_time['h'].astype(int) == 7] 
df_8 = data_time[data_time['h'].astype(int) == 8] 
df_9 = data_time[data_time['h'].astype(int) == 9] 
df_10 = data_time[data_time['h'].astype(int) == 10] 
df_11 = data_time[data_time['h'].astype(int) == 11] 
df_12 = data_time[data_time['h'].astype(int) == 12] 
df_13 = data_time[data_time['h'].astype(int) == 13] 
df_14 = data_time[data_time['h'].astype(int) == 14] 
df_15 = data_time[data_time['h'].astype(int) == 15] 
df_16 = data_time[data_time['h'].astype(int) == 16]

#%%
#b&s
#hr:07
df_7_B = df_7[df_7['買賣別'] == 'B']
df_7_S = df_7[df_7['買賣別'] == 'S']

df_7_B = df_7_B.sort_values(by='委託價', ascending=False)
df_7_S = df_7_S.sort_values(by='委託價', ascending=False)

#del df_7_B['b1','b2','b3','b4','b5']
df_7_B_first = df_7_B.iloc[:5,:]

for i in range(5):
    i = i+1
    df_7_B['b'+i] = df_7_B_first.iloc[i-1,5]

#%%
#b&s
#hr:08
#split to B & S
df_8_B = df_8[df_8['買賣別'] == 'B']
df_8_S = df_8[df_8['買賣別'] == 'S']

#sort
df_8_B = df_8_B.sort_values(by='委託價', ascending=False)
df_8_S = df_8_S.sort_values(by='委託價', ascending=False)

#5-largest value
ar = df_8_B.委託價.unique() #unique value
df_8_B_first = ar[0:5]
df_8_B_first = pd.DataFrame(df_8_B_first)

#fill 5-largest value in df-B
for i in range(5):
    i = i+1
    df_8_B['b'+str(i)] = 0 #initial
    df_8_B['b'+str(i)] = df_8_B_first.loc[i-1,0] #fill
    
    
#%%
def haha(hr):
    dfname = 'df_'+str(hr)
    int(hr)
    dfname = data_time[data_time['h'].astype(int) == hr] 
    
    B,S =[],[]
    B = dfname[dfname['買賣別'] == 'B']
    S = dfname[dfname['買賣別'] == 'S']
    #sort
    B = B.sort_values(by='委託價', ascending=False)
    S = S.sort_values(by='委託價', ascending=False)
    
    #5-largest value
    arB = B.委託價.unique() #unique value
    B_first = arB[0:5]
    B_first = pd.DataFrame(B_first)
    #5-largest value
    arS = S.委託價.unique() #unique value
    S_first = arS[0:5]
    S_first = pd.DataFrame(S_first)
    
    #fill 5-largest value in df-B
    for i in range(5):
        i = i+1
        B['b'+str(i)] = 0 #initial
        B['b'+str(i)] = B_first.loc[i-1,0] #fill
    for i in range(5):
        i = i+1
        S['s'+str(i)] = 0 #initial
        S['s'+str(i)] = S_first.loc[i-1,0] #fill
    
    finname = 'df_'+str(hr)+'_fin'
    finname = pd.concat([S,B], axis=0, ignore_index=False)
    
    return B,S


#%%
x, y =[],[]
x, y = haha(9)
wow = pd.concat([x,y], axis=0, ignore_index=False)
wow = wow.fillna(0)
#%%
w =  data_original["商品代號"].str.startswith("TXF")
w = pd.DataFrame(w)
w.columns = ['TXF']
w = pd.concat([data_original,w],axis = 1, ignore_index=False)      
txf = w[w.TXF == True]  
del txf['TXF']        
#%%
data_2016 = pd.read_csv('wow.txt', sep='\t', header=None)
data_2016.columns = ['OSF_DATE','OSF_PROD_ID','OSF_BS_CODE','OSF_ORDER_QNTY','OSF_UN_QNTY','OSF_ORDER_PRICE','OSF_ORDER_TYPE','OSF_ORDER_COND','OSF_OC_CODE','OSF_ORIG_TIME','OSF_SEQ_NO','OSF_DEL_QNTY','FCM_NO+ORDER_NO+O']                
