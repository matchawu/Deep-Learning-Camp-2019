# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:41:16 2019

@author: ruby
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#data = pd.read_csv('F_lvr_land_A.csv',header=0)
data = pd.read_pickle('Temp_Data.pkl')
data_copy = data.copy()


# OHE 交易標的
from keras.utils import to_categorical
ts_classes = 5

data['交易標的'] = data['交易標的'].map({'土地':0,'車位':1,'房地(土地+建物)':2,'房地(土地+建物)+車位':3,'建物':4}).astype(int)
df_ts = data['交易標的']
df_ts = to_categorical(df_ts, ts_classes)


# 合併 : 都市土地使用分區 + 非都市土地使用分區 = 土地使用分區
df_zone = data[['都市土地使用分區','非都市土地使用分區']]
df_zone = df_zone.fillna("")
df_zone = df_zone.astype(str)

df_zone['都市土地使用分區'] = df_zone['都市土地使用分區'] + df_zone['非都市土地使用分區']
del df_zone['非都市土地使用分區']
df_zone.rename(columns={'都市土地使用分區':''}, inplace=True)


# OHE 土地使用分區 df_zone???
zone_ohe = pd.get_dummies(df_zone) 

# OHE 非都市土地使用編定
df_zone2 = data['非都市土地使用編定']
df_zone2 = df_zone2.fillna(0)
zone2_ohe = pd.get_dummies(df_zone2) 
del zone2_ohe[0]

#合併
land_ohe = pd.concat( [zone_ohe, zone2_ohe], axis=1 )


# OHE 建物型態
bs_ohe = pd.get_dummies(data['建物型態']) 


# 補空格: 將交易標的不是土地OR車位的&&建材為空者 填眾數 (4筆)
df_bm = data['主要建材']
df_bm = df_bm.fillna(-1)

s= set(df_bm)
dictionary = {e:i for i,e in enumerate(s)}
df_bm = df_bm.replace(dictionary)
#眾數
count_bm = df_bm.value_counts()


# 將交易標的不是土地OR車位的&&建材為空者 填眾數 (4筆)
#df_bm = df_bm.astype(int)
#df_bm = pd.DataFrame(df_bm)
df_bm[np.where(df_bm==11)[0]] = 1

# OHE 主要建材 0??
bm_ohe = pd.get_dummies(df_bm) 

#放回去
data = pd.concat([data, df_ts], axis=1, ignore_index=False)
data = pd.concat([data, land_ohe], axis=1, ignore_index=False)
data = pd.concat([data, bs_ohe], axis=1, ignore_index=False)