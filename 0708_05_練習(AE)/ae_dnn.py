# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 18:36:16 2019

@author: wwj
"""
#%%
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model

#%%
(x_train, _),(x_test,_) = mnist.load_data()

#%%
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

#%%
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#%%
#build model
input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train, epochs=10, batch_size=256,shuffle=True,validation_split=0.1)

#%%
#plot result
decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    #original picture
    ax = plt.subplot(2,n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # reconstruction
    ax = plt.subplot(2,n, i+1)
    plt.imshow(decoded_imgs[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
plt.show()

    
