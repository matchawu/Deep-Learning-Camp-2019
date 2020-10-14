# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 18:53:14 2019

@author: wwj
"""
#%%
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

#['dDateTime', 'dCode', 'dClose', 'dOpen', 'dHighest', 'dLowest', 'dVolume']
df = pd.read_csv('TXF20112015_original.csv')

#%%
# delete date but don't shuffle
del df['dDateTime']
del df['dCode']
del df['dVolume']
#%%
random.seed(0)
random_factor = 0.1

#%%
def _load_data(data, n_prev = 30):  
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].values)
        docY.append(data.iloc[i+n_prev].values)
    
    return np.array(docX), np.array(docY)

def train_test_split(df, test_size = 0.2, n_prev = 30):  
    split_point = round(len(df) * (1 - test_size)) 
    X_train, y_train = _load_data(df.iloc[0:split_point], n_prev)
    X_test, y_test = _load_data(df.iloc[split_point:], n_prev)
    
    return (X_train, y_train), (X_test, y_test)

length_of_sequences = 30
(X_train, y_train), (X_test, y_test) = train_test_split(df, n_prev = length_of_sequences)  

#%%
#y_train = pd.DataFrame(y_train)
#y_test = pd.DataFrame(y_test)

#%%
#y_train = y_train.drop([["1", "2", "3","4"]])
#y_test = y_test.drop([["1", "2", "3","4"]])

#%%
from keras.models import Sequential  
from keras.layers.core import Dense  
from keras.layers.recurrent import SimpleRNN, LSTM

inputNode = 4
outputNode = 4
hiddenNode = 10
warmUp = 30

model = Sequential()

model.add(LSTM(hiddenNode, input_shape=(warmUp, inputNode), return_sequences=False))
model.add(Dense(outputNode, activation='relu'))

model.compile(loss='mse', optimizer='adam')
model.summary()

history = model.fit(X_train, y_train, batch_size=128, epochs = 10, validation_split=0.2)


#%%
predicted = model.predict(X_test) 
df_Y_test = pd.DataFrame(Y_test)

#%%
plt.figure(figsize = (18,9))
plt.plot(range(df_Y_test.shape[0]-73700),df_Y_test[73700:],color='b',label='True')
plt.plot(range(np.size(predicted)-73700),predicted[73700:],color='orange',label='Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend(fontsize=18)
plt.show()
#%%
plt.figure(figsize=(10, 5))
plt.plot(predicted[''][:10000], label='predict')
plt.plot(y_test[:10000], label='true label')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('sine wave prediction')
plt.legend()
plt.show()
