# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:11:26 2019

@author: ruby
"""
#%%
import pandas as pd
import numpy as np
#%%
data_all = pd.read_pickle('0704_TempData.pkl')
data = data_all
#data = data_all.copy()

#%%
y = data[['總價元']]
del data['總價元']
#%%
#4 features
#not delete 0707-3
#del data['建物現況格局-房']
#del data['建物現況格局-廳']
#del data['建物現況格局-衛'] 
#del data['建物現況格局-隔間']
#%%
#13 features
#delete 0707-3
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

#%%
#9 features
#delete 0707-3
del data['丁種建築用地']
del data['丙種建築用地']
del data['乙種建築用地']
del data['暫未編定']
del data['林業用地']
del data['水利用地']
del data['特定目的事業用地']
del data['甲種建築用地']
del data['農牧用地']

#%%
#11 features
#not delete 0707-3
#del data['_一般農業區']
#del data['_住']
#del data['_其他']
#del data['_山坡地保育區']
#del data['_商']
#del data['_工']
#del data['_工業區']
#del data['_森林區']
#del data['_特定專用區']
#del data['_農']
#del data['_鄉村區']

#%%
#11 features
#not delete 0707-3
#del data['住宅大樓(11層含以上有電梯)']
#del data['公寓(5樓含以下無電梯)']
#del data['其他']
#del data['套房(1房1廳1衛)']
#del data['工廠']
#del data['店面(店鋪)']
#del data['廠辦']
#del data['華廈(10層含以下有電梯)']
#del data['辦公商業大樓']
#del data['農舍']
#del data['透天厝']

#%%
#26 features
#not delete 0707-3
#del data['三峽區']
#del data['三芝區']
#del data['三重區']
#del data['中和區']
#del data['五股區']
#del data['八里區']
#del data['土城區']
#del data['新店區']
#del data['新莊區']
#del data['板橋區']
#del data['林口區']
#del data['樹林區']
#del data['永和區']
#del data['汐止區']
#del data['泰山區']
#del data['淡水區']
#del data['深坑區']
#del data['瑞芳區']
#del data['石碇區']
#del data['石門區']
#del data['萬里區']
#del data['蘆洲區']
#del data['貢寮區']
#del data['金山區']
#del data['雙溪區']
#del data['鶯歌區']


#%%
#BUILD
from keras.models import Sequential,Model
from keras.layers import Input,Dense
from keras.layers.recurrent import SimpleRNN, LSTM

#%%
#build model
model=Sequential()
#input dimension:8 因為有8個欄位; dense1:node=5
model.add(Dense(units=256,input_dim=65,activation='relu',name='dense_1'))
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

#%%
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
plt.plot(hist.history['val_loss'],color='red')
plt.title('learning_curve(loss)')
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.legend(['train','validation'], loc='best')
plt.show()

#%%
s= set(data['土地區段位置/建物區段門牌'])
dictionary = {e:i for i,e in enumerate(s)}
data['土地區段位置/建物區段門牌'] = data['土地區段位置/建物區段門牌'].replace(dictionary)