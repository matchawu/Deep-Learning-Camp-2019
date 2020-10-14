# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 17:30:28 2019

@author: ruby
"""

from keras.datasets import cifar10
from keras.utils import np_utils
((train_feature, train_label),(test_feature, test_label)) = cifar10.load_data()

#normalization: 每個數值影響力應該一樣
train_feature_vector = train_feature / 255
train_label_onehot = np_utils.to_categorical(train_label)
test_feature_vector = test_feature / 255
test_label_onehot = np_utils.to_categorical(train_label)


from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation

model = Sequential()
model.add(Conv2D(filters=10,
                kernel_size=(3,3),
                input_shape=(32,32,3),
                activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.summary()

#
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#
train_history1 = model.fit(x=train_feature_vector,
                          y=train_label_onehot,
                          validation_split=0.3,
                          epochs=10,
                          batch_size=100,
                          verbose=1)
train_history2 = model.fit(x=train_feature_vector,
                          y=train_label_onehot,
                          validation_split=0.3,
                          epochs=10,
                          batch_size=100,
                          verbose=1)
model.save('model_cifar.h5')
model1 = load_model('model_cifar.h5')
model2 = load_model('model2_cifar.h5')


import matplotlib.pyplot as plt

plt.plot(train_history1.history['acc'])
plt.plot(train_history1.history['val_acc'])
plt.plot(train_history2.history['acc'])
plt.plot(train_history2.history['val_acc'])
plt.title('Train History')
plt.ylabel('acc')
plt.xlabel('Epoch')
plt.legend(['deep_train','deep_validation','shallow_train','shallow_validation'], loc='best')
plt.show()

plt.plot(train_history1.history['loss'])
plt.plot(train_history1.history['val_loss'])
plt.plot(train_history2.history['loss'])
plt.plot(train_history2.history['val_loss'])
plt.title('Train History')
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.legend(['deep_train','deep_validation','shallow_train','shallow_validation'], loc='best')
plt.show()


#顯示預測結果
import numpy as np

prediction = model.predict_classes(test_feature_vector)
idx=100
for i in range(0,5):
    ax = plt.subplot(1,5,1+i)
    ax.imshow(test_feature_vector[idx].reshape((32,32,3)))
    title = "label={}, \n predict={}".format(str(np.argmax(test_label_onehot[idx], axis=0)),
               str(prediction[idx]))
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    idx+=1
plt.show()

#confusion matrix
#import pandas as pd

#print(pd.crosstab(np.argmax(test_label_onehot,axis=1),prediction,rownames=['label'],colnames=['predict']))


#save whole model
model.save('model_cifar.h5')
model = load_model('model_cifar.h5')

