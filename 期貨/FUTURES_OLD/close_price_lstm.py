# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:45:53 2019

@author: wwj
"""
#%%
#if run on server
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))
#%%
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
#%%
#['dDateTime', 'dCode', 'dClose', 'dOpen', 'dHighest', 'dLowest', 'dVolume']
def readTrain():
    train = pd.read_csv('C:\Users\wwj\Desktop\stock\TXF20112015_original.csv')
    return train

#%%
train = readTrain()
del train['dCode']
del train['dDateTime']
del train['dVolume']

#%%
def normalize(train):
    train_norm = train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return train_norm

#%%
# Normalization
train_norm = normalize(train)
#%%
def buildTrain(train, pastDay=30, futureDay=1):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["dClose"]))
    return np.array(X_train), np.array(Y_train)

#%%
# build Data, use last 30 days to predict next 1 days
X_train, Y_train = buildTrain(train_norm, 30, 1)
#%%
def shuffle(X,Y):
    np.random.seed()
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

#%%
# shuffle the data, and random seed is 10
# X_train, Y_train = shuffle(X_train, Y_train)
    
#%%
def splitData(X,Y,rate):
    #0.2-0.8 train
    X_train = X[int(X.shape[0]*rate):int(X.shape[0]*rate*3)]
    Y_train = Y[int(Y.shape[0]*rate):int(Y.shape[0]*rate*3)]
    #0-0.2 validation
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    #0.8-1 test
    X_test = X[int(X.shape[0]*rate*3):]
    Y_test = Y[int(Y.shape[0]*rate*3):]
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

#%%
# split training data and validation data
X_train, Y_train, X_val, Y_val, X_test, Y_test = splitData(X_train, Y_train, 0.3)

#%%
def buildManyToOneModel(shape):
    model = Sequential()
    model.add(LSTM(10, input_length=shape[1], input_dim=shape[2]))
    # output shape: (1, 1)
    model.add(Dense(1))
#     model.add(LeakyReLU(alpha=0.1))
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model

#%%
from keras.models import Sequential
from keras.layers import *
#%%
def ManyToManyScalar(shape):
    model = Sequential()
    model.add(LSTM(100, input_length=shape[1], input_dim=shape[2], dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(7, activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
#from keras.models import Sequential
#from keras.layers import *
#
#model = Sequential()
#model.add(Embedding(5000, 32, input_length=500))
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(7, activation='softmax'))
#model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#model.summary()
#%%
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D,Conv1D  # Convolution Operation
from keras.layers import MaxPooling2D,MaxPooling1D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Reshape

#%%
#LSTM model
model = buildManyToOneModel(X_train.shape)
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
history = model.fit(X_train, Y_train, epochs=30, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])

#%%
# list all data in history
#print(history.history.keys())
#print(history)
#%%
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
#%%
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.legend()
plt.show()
#%%
model.save('my_model_lstm.h5')
#%%
import tensorflow as tf
model = tf.contrib.keras.models.load_model('my_model_lstm.h5')
#%%
score = model.evaluate(X_val, Y_val, verbose=1)
print(score)
#%%
prediction = model.predict(X_test)
print(prediction)
#%%
df_Y_test = pd.DataFrame(Y_test)
print(range(df_Y_test.shape[0]))
#%%
plt.figure(figsize = (18,9))
plt.plot(range(df_Y_test.shape[0]-44000),df_Y_test[44000:],color='b',label='True')
plt.plot(range(np.size(prediction)-44000),prediction[44000:],color='orange',label='Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend(fontsize=18)
plt.show()

#考慮將train, test, original 一起印出來

#%%
w = abs(prediction - df_Y_test)
w.plot()






