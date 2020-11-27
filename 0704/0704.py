# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:11:26 2019

@author: ruby
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_pickle('0704_TempData.pkl')
#data_copy = data.copy()
y = data[['總價元']]
del data['總價元']
del data['建物移轉總面積平方公尺']
del data['屋齡']
del data['平均土地移轉總面積平方公尺'] 
#data_copy =  data_copy.drop(columns=['index'])

#data_top = data.head() 

from sklearn.model_selection import train_test_split
##train:1000 test:last data split
#train_x = data_copy.iloc[:60000,:] #feature
#test_x = data_copy.iloc[60000:,:]
# fare as target
#train_y = y.iloc[:60000] #label
#test_y = y.iloc[60000:]

#%%
#BUILD
#keras
from keras.models import Sequential,Model
from keras.layers import Input,Dense
from keras.layers.recurrent import SimpleRNN, LSTM

#build model
model=Sequential()
#model.add(LSTM(128, input_shape=(512, 92), return_sequences=False))
#input dimension:8 因為有8個欄位; dense1:node=5
model.add(Dense(units=512,input_dim=84,activation='relu',name='dense_1'))
model.add(Dense(units=256,activation='relu',name='dense_2'))
model.add(Dense(units=128,activation='relu',name='dense_3'))
model.add(Dense(units=128,activation='relu',name='dense_4'))
model.add(Dense(units=64,activation='relu',name='dense_5'))
#output dimension:1 只有船票一個值，故為1
model.add(Dense(units=1,activation='relu',name='output_layer'))

#顯示model的架構參數量
model.summary()


#COMPILE: mape沒有acc
model.compile(loss='MAPE',
              optimizer='adam',
              metrics=['accuracy'])
#fit
hist = model.fit(data,y,
          batch_size=64,
          epochs=50,
          validation_split=0.2,
          verbose=1)
#%%
#paint
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],color='cyan')
plt.plot(hist.history['val_loss'],color='magenta')
plt.title('learning_curve(loss)')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['train','validation'], loc='best')
plt.show()

#plt.plot(hist.history['acc'])
#plt.plot(hist.history['val_acc'])
#plt.title('learning_curve(acc)')
#plt.xlabel('epochs')
#plt.ylabel('Accuracy')
#plt.legend(['train','validation'], loc='best')
#plt.show()

#%%
s= set(data['土地區段位置/建物區段門牌'])
dictionary = {e:i for i,e in enumerate(s)}
data['土地區段位置/建物區段門牌'] = data['土地區段位置/建物區段門牌'].replace(dictionary)