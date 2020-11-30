# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 22:41:25 2019

@author: wwj
"""
#%%
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
#%%
#['dDateTime', 'dCode', 'dClose', 'dOpen', 'dHighest', 'dLowest', 'dVolume']
def readTrain():
  train = pd.read_csv('TXF20112015_original.csv')
  return train

# read SPY.csv
train = readTrain()
del train['dCode']
#%%
def augFeatures(train):
  train["year"] = train["dDateTime"]//100000000
  train["month"] = (train["dDateTime"]%100000000)//1000000
  train["date"] = (train["dDateTime"]%1000000)//10000
  train["hour"] = (train["dDateTime"]%10000)//100
  train["min"] = train["dDateTime"]%100
  return train

# Augment the features (year, month, date, day)
train_Aug = augFeatures(train)
#%%
#del train_Aug['dDateTime']
#%%
def normalize(train):
  train = train.drop(["dDateTime"], axis=1)
  train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train_norm

# Normalization
train_norm = normalize(train_Aug)

#%%
def buildTrain(train, pastDay=30, futureDay=1):
  X_train, Y_train = [], []
  for i in range(train.shape[0]-futureDay-pastDay):
    X_train.append(np.array(train.iloc[i:i+pastDay]))
    Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["dClose"]))
  return np.array(X_train), np.array(Y_train)

# build Data, use last 30 days to predict next 1 days
X_train, Y_train = buildTrain(train_norm, 30, 1)
#%%
def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]

# shuffle the data, and random seed is 10
X_train, Y_train = shuffle(X_train, Y_train)
#%%
def splitData(X,Y,rate):
    #0.2-0.8 train
  X_train = X[int(X.shape[0]*rate):int(X.shape[0]*rate*4)]
  Y_train = Y[int(Y.shape[0]*rate):int(Y.shape[0]*rate*4)]
    #0-0.2 validation
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
    #0.8-1 test
  X_test = X[int(X.shape[0]*rate*4):]
  Y_test = Y[int(Y.shape[0]*rate*4):]
  return X_train, Y_train, X_val, Y_val, X_test, Y_test

# split training data and validation data
X_train, Y_train, X_val, Y_val, X_test, Y_test = splitData(X_train, Y_train, 0.2)

#%%
def buildManyToOneModel(shape):
  model = Sequential()
  model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
  # output shape: (1, 1)
  model.add(Dense(1))
  model.compile(loss="mse", optimizer="adam")
  model.summary()
  return model

#%%
#train = readTrain()
#train_Aug = augFeatures(train)
#train_norm = normalize(train_Aug)

# change the last day and next day 
#X_train, Y_train = buildTrain(train_norm, 30, 1)
#X_train, Y_train = shuffle(X_train, Y_train)

# because no return sequence, Y_train and Y_val shape must be 2 dimension
#X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.1)

#%%
model = buildManyToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
history = model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])
#%%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()
#%%
model.save('my_model.h5')
#%%
import tensorflow as tf
model = tf.contrib.keras.models.load_model('my_model_01.h5')
score = model.evaluate(X_val, Y_val, verbose=1)
print(score)
#%%
prediction = model.predict(X_test)
df_Y_test = pd.DataFrame(Y_test)
#%%
plt.figure(figsize = (18,9))
plt.plot(range(df_Y_test.shape[0]),df_Y_test,color='b',label='True')
plt.plot(range(np.size(prediction)),prediction,color='orange',label='Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend(fontsize=18)
plt.show()

















