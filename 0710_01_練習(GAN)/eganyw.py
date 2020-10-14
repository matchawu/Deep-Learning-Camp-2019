# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 10:40:08 2019

@author: yawen
"""

import numpy as np

import matplotlib.pyplot as plt

from glob import glob

import random

from PIL import Image

from keras.models import Model

from keras.optimizers import Adam

from keras.layers import Conv2D, UpSampling2D, Dense, Flatten, Input,BatchNormalization, Reshape, LeakyReLU
#%%
class DataLoader:

    def __init__(self, folder_path, img_size):
    
        self.folder_path = folder_path
        
        self.img_size=img_size
        
        self.img_size = img_size
        #glob把整個資料夾的東西讀進來
        self.path_list = glob(self.folder_path)
        # 讀取資料夾全部圖片路徑
        assert len(self.path_list)>0, 'error'
    def __imread(self, img_path):
    
        '''讀取圖片'''
    
        return np.array(Image.open(img_path).convert('RGB').resize(self.img_size[:-1],Image.ANTIALIAS))
    
    def sampling_data(self, batch_size, shuffle=True):
        #一次要抽多少張出來
        #打亂
        img_path_list = self.path_list
        
        if shuffle:
        
            random.shuffle(img_path_list)
        
        for batch_idx in range(0, len(img_path_list), batch_size):

            path_set = img_path_list[batch_idx : batch_idx + batch_size]
            
            img_set = np.zeros((len(path_set),) + self.img_size)
        
            for img_idx, path in enumerate(path_set):
            
                img_set[img_idx] = self.__imread(path)
                
            img_set = img_set/127.5 -1#把圖片壓成-1到1
            
            yield img_set #把這個fun暫停而不是結束
#%%
data_loader=DataLoader('./cartoon/cartoon/*.png',(32,32,3))      
a=next(data_loader.sampling_data(1))
import matplotlib.pyplot as plt
plt.imshow(a[0]*.5+.5)

#%%
class GAN:

    def __init__(self, noise_dim, img_size):
    
        self.noise_dim = noise_dim
        
        self.img_size = img_size
        
        self.dataloader = DataLoader('./cartoon/cartoon/*.png', self.img_size)
        
        # noise_dim = 雜訊維度
        
        # img_size = 圖片大小

    def build_generator(self):
        
        
        noise_input = Input(shape=(self.noise_dim,))
        h = Dense(1024)(noise_input)
        h = LeakyReLU(alpha=0.2)(h) 
        h = Dense(1024)(h)     
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)     
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(256)(h)     
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(32*32*3,activation='tanh')(h) 
        img = Reshape(self.img_size)(h)
        # TODO: Build generator
        
        return Model(noise_input, img)

    def build_discriminator(self):
    
        img_input = Input(shape=self.img_size)
        h = Flatten()(img_input) 
        h = Dense(128)(h) 
        h = LeakyReLU(alpha=0.2)(h)        
        h = Dense(64)(h)     
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(64)(h)     
        h = LeakyReLU(alpha=0.2)(h)
        validity = Dense(1,activation='sigmoid')(h) 
        # TODO: Build generator
        
        return Model(img_input, validity)

    def connect(self):

        self.generator = self.build_generator()
        print(self.generator.count_params())
        self.generator.summary()        
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()        
        print(self.discriminator.count_params())
        
        self.d_optimizer = Adam(.0002, .5)
        self.g_optimizer = Adam(.0002, .5)

        # Optimizer用Adam, Learning rate=0.0001~0.0002, 切勿調高

        self.discriminator.compile(optimizer=self.d_optimizer,

                                    loss='binary_crossentropy',
                                    
                                    metrics=['acc'])
        # 連接G和D

        noise = Input(shape=(self.noise_dim,))
        
        img = self.generator(noise)
        
        self.discriminator.trainable = False
        # 在訓練G時, 鎖定D
        
        validity = self.discriminator(img) #0~1的分數
        
        self.combined = Model(noise, validity)
        
        self.combined.compile(optimizer=self.g_optimizer,
                              loss='binary_crossentropy')

    def train(self, epochs, batch_size, sample_interval):
    
        self.history = []
        self.sample_interval=sample_interval
        valid = np.ones((batch_size, 1))
        # 1 = 真實圖片
        
        fake = np.zeros((batch_size, 1))
        # 0 = 生成圖片
        for e in range(epochs): 
            for i, real_img in enumerate(self.dataloader.sampling_data(batch_size)):          
                # Train D               
                noise = np.random.standard_normal((batch_size, self.noise_dim))
                
                fake_img = self.generator.predict(noise)
                
                d_loss_real, real_acc = self.discriminator.train_on_batch(real_img, valid[:len(real_img)])
                
                d_loss_fake, fake_acc = self.discriminator.train_on_batch(fake_img, fake)
                
                d_loss = .5 * (d_loss_real + d_loss_fake)
                
                d_acc =  .5 * (real_acc + fake_acc)
        
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
                    
                    print('[Epoch %(epoch)d][Iteration %(iter)d][D loss: %(d_loss).6f, acc: %(d_acc).2f][G loss: %(g_loss).6f]' % info)
                    
            self.__sample_image(e)
                    
        return self.history

    def __sample_image(self, epoch):
    
        r, c = 8, 8
        # 列, 欄
        
        noise = np.random.standard_normal((r*c, self.noise_dim))
        
        img = self.generator.predict(noise).reshape((r, c) + self.img_size)
        
        img = img * .5 + .5
        
        fig = plt.figure(figsize=(20, 20))
        
        axs = fig.subplots(r, c)
        
        for i in range(r):
        
            for j in range(c):
            
                axs[i, j].imshow(img[i, j])
                
                axs[i, j].axis('off')
                
                fig.savefig('images/gan8/%05d.png' % epoch)
                
                plt.close()
 #%%
#img.summary()
#validity.summary()

 #%%
gan=GAN(128,(32,32,3))
gan.connect()
gan.train(200, 64,10)
