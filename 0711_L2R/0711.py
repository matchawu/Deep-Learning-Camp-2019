# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:24:51 2019

@author: wwj
"""

#%%
import pandas as pd
import numpy as np
#%%
data = pd.read_csv('ranking.csv')
data_copy = data.copy()

#%%
del data_copy['performance_points']
del data_copy['rank']

#%%
device = data_copy['device']
device = pd.get_dummies(device)

#%%
del data_copy['device']

copy = pd.concat([data_copy,device],axis=0, ignore_index=True)
