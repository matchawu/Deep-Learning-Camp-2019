# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:12:57 2019

@author: ruby
"""

from keras.datasets import mnist
from keras.utils import np_utils

#read mnist data
(X_Train,y_Train),(X_Test,y_Test)=mnist.load_data()

#translation
X_Train = X_Train.reshape(X_Train.shape[0],28,28,1).astype('float32')
X_Test = X_Test.reshape(X_Test.shape[0],28,28,1).astype('float32')

#standardize
X_Train = X_Train /255
X_Test = X_Test/255

#OHE
y_Train = np_utils.to_categorical(y_Train)
y_Test = np_utils.to_categorical(y_Test)


from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPooling2D

model = Sequential()
#create CN layer 1
model.add(Conv2D(filters=5,kernel_size=(3,3),input_shape=(28,28,1),activation='relu'))
#create Max-Pool 1
model.add(MaxPooling2D(pool_size=(2,2)))

#create CN layer 2
model.add(Conv2D(filters=10,kernel_size=(3,3),activation='relu'))
#create Max-Pool 2
model.add(MaxPooling2D(pool_size=(2,2)))

#create CN layer 3
model.add(Conv2D(filters=15,kernel_size=(3,3),activation='relu'))
#create Max-Pool 3
model.add(MaxPooling2D(pool_size=(2,2)))

#2D to 1D
model.add(Flatten()) 
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history1 = model.fit(x=X_Train, y=y_Train, 
                        validation_split=0.2, 
                        epochs=10, 
                        batch_size=300, 
                        verbose=1)

import matplotlib.pyplot as plt
import numpy as np
print('mean acc:',np.mean(train_history1.history["acc"]))
plt.plot(train_history1.history['acc'])
plt.plot(train_history1.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc','val_acc'],loc='lower right')
plt.show()

print('mean loss:',np.mean(train_history1.history["loss"]))
plt.plot(train_history1.history['loss'])
plt.plot(train_history1.history['val_loss'])
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_loss','val_loss'],loc='upper right')
plt.show()

prediction = model.predict_classes(X_Test)
idx = 100
for i in range(0,5):
    ax = plt.subplot(1,5,1+i)
    ax.imshow(X_Test[idx].reshape((28,28)),cmap='gray')
    title = "1={},p={}".format(str(np.argmax(y_Test[idx],axis=0)),
               str(prediction[idx]))
    ax.set_title(title,fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    idx+=1
plt.show()

#顯示confusion matrix
import pandas as pd

print(pd.crosstab(np.argmax(y_Test,axis=1),prediction,rownames=['label'],colnames=['predict']))









