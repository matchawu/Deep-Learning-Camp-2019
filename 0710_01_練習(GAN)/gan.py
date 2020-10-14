# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:36:01 2019

@author: wwj
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import random
from PIL import Image
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Conv2D, UpSampling2D, Dense, Flatten, Input, BatchNormalization, Reshape, LeakyReLU
from keras.models import Sequential
from keras.layers import Dense, Activation
#%%
#class > def def def
#讀取資料
class DataLoader:
    def __init__(self, folder_path, img_size):
        self.folder_path = folder_path
        self.img_size = img_size
        
        self.path_list = glob(folder_path)		# 讀取資料夾全部圖片路徑
    
    def __imread(self, img_path):
            
        return np.array(Image.open(img_path).convert('RGB').resize(self.img_size[:-1]))


    def sampling_data(self, batch_size, shuffle=True):
        img_path_list = self.path_list
        
        if shuffle:
            random.shuffle(img_path_list)


        for batch_idx in range(0, len(img_path_list), batch_size):
            path_set = img_path_list[batch_idx : batch_idx + batch_size]
            img_set = np.zeros((len(path_set),) + self.img_size)
            for img_idx, path in enumerate(path_set):
                img_set[img_idx] = self.__imread(path)
            img_set = img_set / 127.5 - 1
            yield img_set
#%%
dataloader = DataLoader('./cartoon/cartoon/*.png',(32,32,3))
a = next(dataloader.sampling_data(32))
#import matplotlib.pyplot as plt
plt.imshow(a[0]*.5+.5) #plt要求在0~1之間

#%%
#建立class
class GAN:
    def __init__(self, noise_dim, img_size=(64, 64, 3)):
        #gan會有的參數 
        self.noise_dim = noise_dim
        self.img_size = img_size
        self.dataloader = DataLoader('./cartoon/cartoon/*.png', self.img_size)
        # noise_dim = 雜訊維度
        # img_size = 圖片大小
    def build_generator(self):
        noise_input = Input(shape=(self.noise_dim,))
        # TODO: Build generator
        h = Dense(256)(noise_input)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(256)(h)
#        h = BatchNormalization()
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(1024)(h)
#        h = BatchNormalization()
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(32*32*3, activation='tanh')(h)
        img = Reshape(self.img_size)(h)
        return Model(noise_input, img)
    
    def build_discriminator(self):
        img_input = Input(shape=self.img_size)
        # TODO: Build discriminator
        h = Flatten()(img_input)
        h = Dense(64)(h)
        h = LeakyReLU(alpha=0.2)
        h = Dense(64)(h)
        h = LeakyReLU(alpha=0.2)
        h = Dense(64)(h)
        h = LeakyReLU(alpha=0.2)
        validity = Dense(1, activation='sigmoid')(h)
        return Model(img_input, validity)
    
    
    def connect(self):
        #連接網路
        self.generator = self.build_generator()
        print(self.generator.count_params())
        self.discriminator = self.build_discriminator()
        print(self.discriminator.count_params())
        self.optimizer = Adam(.0002, .5)
        #直接打adam會壞掉
        # Optimizer用Adam, Learning rate=0.0001~0.0002, 切勿調高
        #0,1 binary cross entropy
        self.discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['acc'])
        
        # 連接G和D
        noise = Input(shape=(self.noise_dim,))
        img = self.generator(noise)
        self.discriminator.trainable = False	# 在訓練G時, 鎖定D
        validity = self.discriminator(img) # 0~1分數
        self.combined = Model(noise, validity)
        self.combined.compile(optimizer=self.optimizer, loss='binary_crossentropy') #間接訓練到G
        
    def train(self, epochs, batch_size, sample_interval=200):
        self.history = []
        valid = np.ones((batch_size, 1))		# 1 = 真實圖片
        fake = np.zeros((batch_size, 1))		# 0 = 生成圖片
        for e in range(epochs):
            for i, real_img in enumerate(self.dataloader.sampling_data(batch_size)):
                # Train D
                noise = np.random.standard_normal((batch_size, self.noise_dim))
                fake_img = self.generator.predict(noise)
                
                d_loss_real, real_acc = self.discriminator.train_on_batch(real_img, valid[:len(real_img)])
                d_loss_fake, fake_acc = self.discriminator.train_on_batch(fake_img, fake)
                d_loss = .5 * (d_loss_real + d_loss_fake)
                d_acc = .5 * (real_acc + fake_acc)
                
                # Train G
                noise = np.random.standard_normal((batch_size, self.noise_dim))
                g_loss = self.combined.train_on_batch(noise, valid)
                
                if i % sample_interval == 0:
                    info = {
                            'epoch': e,
                            'iter': i,
                            'd_loss': d_loss,
                            'd_acc': d_acc*100,
                            'g_loss': g_loss
                            }
                    self.history.append(list(info.values()))
                    print('[Epoch %(epoch)d][Iteration %(iter)d] [D loss: %(d_loss).6f, acc: %(d_acc).2f%%] [G loss: %(g_loss).6f]' % info)
            __sample_image(e)
        return self.history
    def __sample_image(self, epoch):
        r, c = 8, 8		# 列, 欄
        noise = np.random.standard_normal((r*c, self.noise_dim))
        img = self.generator.predict(noise).reshape((r, c) + self.img_size)
        img = img * .5 + .5
        fig = plt.figure(figsize=(20, 20))
        axs = fig.subplots(r, c)
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(img[i, j])
                axs[i, j].axis('off')
        fig.savefig('../Image/%05d.png' % epoch)
        plt.close()

#%%
#gan = GAN(100, img_size=(32,32,3))
#gan.connect()

#%%
gan = GAN(128, img_size=(32, 32, 3))
gan.connect()
gan.train(200, 64, sample_interval=10)









