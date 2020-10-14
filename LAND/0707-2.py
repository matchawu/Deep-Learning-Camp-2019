# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:11:26 2019

@author: ruby
"""
import pandas as pd
import numpy as np
data = pd.read_pickle('0704_TempData.pkl')
data_all = pd.read_pickle('Temp_Data.pkl')
#data_copy = data.copy()
y = data[['總價元']]
del data['總價元']


del data['建物現況格局-房']
del data['建物現況格局-廳']
del data['建物現況格局-衛'] 
del data['建物現況格局-隔間']
del data['鋼骨鋼筋混凝土造']
del data['鋼骨混凝土造']
del data['磚造']
del data['鋼筋混凝土加強磚造']
del data['見使用執照']
del data['土木造']
del data['鋼筋混凝土造']
del data['加強磚造']
del data['土磚石混合造']
del data['鋼造']
del data['石造']
del data['見其他登記事項']
del data['土造']
del data['丁種建築用地']
del data['丙種建築用地']
del data['乙種建築用地']
del data['暫未編定']
del data['林業用地']
del data['水利用地']
del data['特定目的事業用地']
del data['甲種建築用地']
del data['農牧用地']

del data['_一般農業區']
del data['_住']
del data['_其他']
del data['_山坡地保育區']
del data['_商']
del data['_工']
del data['_工業區']
del data['_森林區']
del data['_特定專用區']
del data['_農']
del data['_鄉村區']

#from sklearn.model_selection import train_test_split
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
model.add(Dense(units=256,input_dim=50,activation='relu',name='dense_1'))
model.add(Dense(units=128,activation='relu',name='2'))
model.add(Dense(units=64,activation='relu',name='3'))
model.add(Dense(units=64,activation='relu',name='4'))
model.add(Dense(units=64,activation='relu',name='5'))
model.add(Dense(units=64,activation='relu',name='6'))
model.add(Dense(units=64,activation='relu',name='7'))
model.add(Dense(units=64,activation='relu',name='8'))
model.add(Dense(units=64,activation='relu',name='9'))
model.add(Dense(units=64,activation='relu',name='10'))
model.add(Dense(units=64,activation='relu',name='11'))
model.add(Dense(units=64,activation='relu',name='12'))
model.add(Dense(units=64,activation='relu',name='13'))
model.add(Dense(units=64,activation='relu',name='14'))
model.add(Dense(units=32,activation='relu',name='15'))

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
          epochs=200,
          validation_split=0.2,
          verbose=1)
#%%
#paint
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],color='cyan')
plt.plot(hist.history['val_loss'],color='red')
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