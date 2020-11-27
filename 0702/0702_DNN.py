# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:29:12 2019

@author: ruby
"""
import numpy as np
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#畫法一
#import matplotlib.pyplot as plt
#n = 3 #畫三張圖
#plt.figure(figsize=(n*2, 2)) #圖size
#for i in range(n):
#    ax = plt.subplot(1, n, i+1)
#    plt.imshow(x_train[i].reshape(28, 28))
#    plt.gray()
#    ax.get_xaxis().set_visible(False)
#    ax.get_yaxis().set_visible(False)
#plt.show()
#畫法二
#import matplotlib.pyplot as plt
#plt.imshow(x_train[0],cmap='winter')
#plt.show()
#plt.imshow(x_train[1],cmap='cool')
#plt.show()

#切資料
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#值域標準化
x_train /= 255
x_test /= 255

#ONE HOT ENCODING
from keras.utils import to_categorical
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#BUILD
from keras.layers import Input, Dense
from keras.models import Model

input_x = Input(shape=(784,))
x = Dense(20, activation='relu')(input_x)
x1 = Dense(20, activation='relu')(x)
output_y = Dense(10, activation='softmax')(x1)

model = Model(inputs = input_x, outputs = output_y)

#COMPILE
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#fit
hist = model.fit(x_train, y_train, 
          batch_size=512,
          epochs=20,
          validation_split=0.2,
          verbose=1)

#paint
import matplotlib.pyplot as plt
plt.plot(hist.history['loss'],color='cyan')
plt.plot(hist.history['val_loss'],color='magenta')
plt.title('learning_curve(loss)')
plt.xlabel('epochs')
plt.ylabel('categorical_crossentropy')
plt.legend(['train','validation'], loc='best')
plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('learning_curve(acc)')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.legend(['train','validation'], loc='best')
plt.show()

#在評價模型訓練成效時，需要了解模型在測試資料上的表現
model_pred = model.predict(x_test)
model_ans = np.argmax(model_pred, axis=1)
#因為前面坐了ohe
real_ans = np.argmax(y_test, axis=1)
#(model_ans-real_ans)==0 代表猜對
test_acc = np.mean((model_ans-real_ans)==0)
print('accuracy：',test_acc)
#混淆矩陣(confusion matrix)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(real_ans, model_ans)
