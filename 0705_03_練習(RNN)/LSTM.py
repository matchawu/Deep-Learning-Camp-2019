# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:55:25 2019

@author: ruby
"""
#%%
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(0)
#平滑程度
random_factor = 0.1


steps_per_cycle = 80 
number_of_cycles = 200

df = pd.DataFrame(np.arange(steps_per_cycle * number_of_cycles), columns = ["t"])
df["sin_t"] = df.t.apply(lambda x: math.sin(x * (2 * math.pi / steps_per_cycle) + random.uniform(-1.0, +1.0) * random_factor))

def _load_data(data, n_prev = 100):  
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].values)
        docY.append(data.iloc[i+n_prev].values)
    
    return np.array(docX), np.array(docY)

def train_test_split(df, test_size = 0.1, n_prev = 100):  
    split_point = round(len(df) * (1 - test_size)) 
    X_train, y_train = _load_data(df.iloc[0:split_point], n_prev)
    X_test, y_test = _load_data(df.iloc[split_point:], n_prev)
    
    return (X_train, y_train), (X_test, y_test)

#看100預測101
length_of_sequences = 100
(X_train, y_train), (X_test, y_test) = train_test_split(df[["sin_t"]], n_prev = length_of_sequences)  

#看平滑程度
plt.plot(df[['sin_t']][:100])
plt.show()

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential  
from keras.layers.core import Dense  
from keras.layers.recurrent import SimpleRNN, LSTM

#每次一筆 送100次
inputNode = 1
#only one scalar
outputNode = 1
#memory 數量
hiddenNode = 10
#要看前面100個 預測第101個
warmUp = 100

model = Sequential() 

#input_shape = (batch_size512 //keras會自動給LSTM不用寫出來,timestep100,feature dimension 1)
#return_sequences 預設到最後一個timestep才會吐東西 (False)
#true: (512,100,10層)
#false: (512,10) 如果接多層LSTM 會錯 因為到下一層需要三維
model.add(LSTM(hiddenNode, input_shape=(warmUp, inputNode), return_sequences=False))
model.add(Dense(outputNode, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.summary()

history = model.fit(X_train, y_train, batch_size=128, epochs = 10, validation_split=0.2)

#print(X_train.shape)
#print(y_train.shape)


#%%

predicted = model.predict(X_test) 

plt.figure(figsize=(10, 5))
plt.plot(predicted[:150], label='predict')
plt.plot(y_test[:150], label='true label')
plt.xlabel('t')
plt.ylabel('x(t)')
plt.title('sine wave prediction')
plt.legend()
plt.show()
