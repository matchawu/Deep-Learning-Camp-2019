# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 10:51:40 2019

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

train = readTrain()
del train['dCode']
del train['dDateTime']
del train['dVolume']

#%%
def normalize(train):
    train_norm = train.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return train_norm

# Normalization
train_norm = normalize(train)
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
    np.random.seed()
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]

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
from keras.models import Sequential  #用來啟動 NN
from keras.layers import Conv2D,Conv1D  # Convolution Operation
from keras.layers import MaxPooling2D,MaxPooling1D # Pooling
from keras.layers import Flatten
from keras.layers import Dense # Fully Connected Networks
from keras.layers import Reshape

#%%
# define 1d CNN model
model_CNN = Sequential()
model_CNN.add(Conv1D(filters=16, kernel_size=2, input_shape=(30, 4)))
model_CNN.add(LeakyReLU(alpha=0.1))
model_CNN.add(Dropout(0.5))
model_CNN.add(MaxPooling1D(pool_size=2))
model_CNN.add(Flatten())
model_CNN.add(Dense(1))
model_CNN.add(LeakyReLU(alpha=0.1))

model_CNN.summary()
#%%
model_CNN.compile(optimizer='adam', loss='mse')
# fit model
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
history_CNN = model_CNN.fit(X_train, Y_train, epochs=3000, batch_size=512, validation_data=(X_val, Y_val), verbose=1,callbacks=[callback])
#%%
# list all data in history_CNN
print(history_CNN.history.keys())
plt.plot(history_CNN.history['loss'])
plt.plot(history_CNN.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#%%
# demonstrate prediction
# x_input = array([[70,75,145,30], [80,85,165,78], [90,95,185,65]])
# X_test_input = X_test.reshape((1, 30, 4))
PRED_CNN = model_CNN.predict(X_test, verbose=1)
print(PRED_CNN)
#%%
Y_test = pd.DataFrame(Y_test)
plt.figure(figsize = (18,9))
plt.plot(range(df_Y_test.shape[0]-44000),df_Y_test[44000:],color='b',label='True')
plt.plot(range(np.size(PRED_CNN)-44000),PRED_CNN[44000:],color='orange',label='Prediction')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend(fontsize=18)
plt.show()


























