# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 14:17:40 2019

@author: ruby
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#load
accepted = pd.read_excel('ICLR_accepted.xlsx')
rejected = pd.read_excel('ICLR_rejected.xlsx')

accepted.rename(columns={'Unnamed: 0':'del'}, inplace=True)
rejected.rename(columns={'Unnamed: 0':'del'}, inplace=True)
del accepted['del']
del rejected['del']
accepted_label = np.ones((582))
rejected_label = np.zeros((753))


accepted_label = pd.DataFrame(accepted_label)
rejected_label = pd.DataFrame(rejected_label)
#accepted_label.rename(columns={'0':'accepted'}, inplace=True)
#rejected_label.rename(columns={'0':'accepted'}, inplace=True)

accepted = pd.concat([accepted, accepted_label], axis=1, ignore_index=True)
rejected = pd.concat([rejected, rejected_label], axis=1, ignore_index=True)
#spilt
accepted_train = accepted.iloc[50:,:]
accepted_test = accepted.iloc[:50,:]
rejected_train = rejected.iloc[50:,:]
rejected_test = rejected.iloc[:50,:]


train_all = pd.concat([accepted_train, rejected_train], axis=0, ignore_index=True)
all_train = pd.concat([accepted_train, rejected_train], axis=0, ignore_index=True)
all_rejected = pd.concat([accepted_test, rejected_test], axis=0, ignore_index=True)
from sklearn.utils import shuffle
train_all = shuffle(train_all)
all_train = shuffle(all_train)
all_rejected = shuffle(all_rejected)
x_train = all_train[0]
y_train = all_train[1]
x_test = all_rejected[0]
y_test = all_rejected[1]
#lower case
train_all[[0]] = train_all[[0]].apply(lambda x : x.str.lower())

#split
train_all[[0]] = train_all[[0]].apply(lambda x : x.str.split(' '))

data = train_all
del data[1]

#data = data.values
#set
#s= set(df_bm)
#dictionary = {e:i for i,e in enumerate(s)}
#df_bm = df_bm.replace(dictionary)

#s = set(data)
#dictionary = {e:i for i,e in enumerate(s)}
#%%
data_t = tuple(data[0])
data0 = []
for i in range(1235):
    temp = data_t[i]
    for j in temp:
        data0.append(j)
#%%
s = set(data0)
dictionary = {e:i for i,e in enumerate(s)}

data = data.reset_index(drop=True)

#for i in range(1235):
#    data.iloc[i] = data.iloc[i].replace(dictionary)
train = np.zeros((1235, 10))
for idx in range(1235):
    temp = data.iloc[idx]
    k=[]
    for j in (temp[0][:]):
        k.append(dictionary[j])
    if len(k)<10:
        for i in range((10-len(k))):
            k.append(0)
    if len(k)>10:
        k = k[:10]
    train[idx] = k

#%%
#other
#dictionary['key'] = value
y_train = np.array(y_train)
#y_train = y_train.index_reset()
#%%
from keras.models import Sequential,Model
from keras.layers import Input,Dense
from keras.layers.recurrent import SimpleRNN, LSTM

from keras.layers import Embedding


model = Sequential() 
model.add(Embedding(2421, 2421, input_length=10))
model.add(LSTM(input_shape=(10, 2421), units = 8, return_sequences=False))
model.add(Dense(2421, activation='linear'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

history = model.fit(train, y_train, batch_size=128, epochs = 10, validation_split=0.2)

#%%
predicted = model.predict(x_test) 

plt.figure(figsize=(10, 5))
plt.plot(predicted[:150], label='predict')
plt.plot(y_test[:150], label='true label')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.legend()
plt.show()























