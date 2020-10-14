# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 16:40:27 2019

@author: wwj
"""

# coding: utf-8
# 可參考 https://keras.io/examples/variational_autoencoder/

#%%
# TODO: import你會用到的套件

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os

from keras.losses import binary_crossentropy
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Input, BatchNormalization, Flatten, Lambda, Reshape, LeakyReLU, ReLU



#%%
img_path = './cartoon'
img_rows = 32
img_cols = 32
channels = 3
img_size = (img_rows, img_cols, channels)
img_path_list = glob(img_path + '/*/*.png')


#%%
# TODO: 讀圖片檔(會用到RGB、resize、normalize)
def load_data( img_res, path_list):
    def imread(path, img_res):
        return np.array(Image.open(path).convert('RGB').resize(img_res, Image.ANTIALIAS)) / 255
    img_set  = np.zeros((len(path_list),) + img_res)
    for idx, rand_img_path in enumerate(path_list):
        img_set[idx] = imread(rand_img_path, img_res[:2])[:,:,:3]
        return img_set
img_data = load_data(img_size, img_path_list)

def build_encoder():
    # TODO: 建 sampling 方式
    def sampling(args):
        z_mu, z_logvar = args
        batch = K.shape(z_mu)[0]
        dim = K.int_shape(z_mu)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mu + K.exp(0.5 * z_logvar) * epsilon
    
    # TODO: 建encoder     
    # build encoder model
    inputs = Input(shape=img_size, name='encoder_input')        
    h = Conv2D(16, kernel_size=4, strides=2, padding='same', name='conv1')(inputs)
    h = LeakyReLU(alpha=0.2)(h)
    
    h = Conv2D(32, kernel_size=4, strides=2, padding='same', name='conv2')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.2)(h)
    
    h = Conv2D(64, kernel_size=4, strides=2, padding='same', name='conv3')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.2)(h)
    
    h = Conv2D(128, kernel_size=4, strides=2, padding='same', name='conv4')(h)
    h = BatchNormalization()(h)
    h = ReLU()(h)
    
    h = Flatten()(h)
            
    z_mu = Dense(latent_dim, name='z_mu')(h)
    z_logvar = Dense(latent_dim, name='z_logvar')(h)
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mu, z_logvar])
    
    return Model(inputs, [z, z_mu, z_logvar], name='encoder')

#%%
# TODO: 建decoder
def build_decoder():
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    h = Reshape((1, 1, latent_dim))(latent_inputs)
    h = Conv2DTranspose(128, kernel_size=4, strides=1, padding='valid', name='convT1')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.2)(h)
    
    h = Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', name='convT2')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.2)(h)

    h = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same', name='convT3')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.2)(h)
    
    h = Conv2DTranspose(16, kernel_size=4, strides=2, padding='same', name='convT4')(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.2)(h)
    
    outputs = Conv2DTranspose(3, kernel_size=4, strides=1, padding='same', activation='sigmoid', name='convT5')(h)

    return Model(latent_inputs, outputs, name='decoder')

#%%
# TODO: 畫出重構圖片   
def sample_image(iteration):
    r, c = 5, 4
#    r, c = 4,3 
    
    idx = np.random.choice(len(img_data), r*c, replace=False)# 隨機選r*c張
    y = combined.predict(img_data[idx])# model predict

    fig, axs = plt.subplots(r, c, figsize=(10,10))
    cnt = 0
    for j in range(r):
        for i in range(c):
            show = np.zeros((img_size[0], img_size[1]*2, img_size[2]))
            show[:,:img_size[1]] = img_data[idx[cnt]]
            show[:,img_size[1]:] = y[cnt]
            axs[j, i].imshow(show)
            axs[j, i].axis('off')
            cnt = cnt+1
    fig.savefig("images/reconst/%d.png" % iteration)
    plt.close()

#%%
# TODO: 畫出亂數生成圖片 
def generate_image(iteration):
    r, c = 5, 5
#    r, c = 4,4
    z = np.random.standard_normal((r*c, latent_dim))
    gen_img = decoder.predict(z)
    
    fig, axs = plt.subplots(r, c, figsize=(10,10))
    cnt = 0
    for j in range(r):
        for i in range(c):                
            axs[j, i].imshow(gen_img[cnt])
            axs[j, i].axis('off')
            cnt = cnt+1
    fig.savefig("images/generate/%d.png" % iteration)
    plt.close()

#%%
# TODO: 畫出morphing生成圖片(內插法)
def generate_morphing_image(model):
    r, c = 6, 6
#    r, c = 7,7
    z = np.zeros((r*c, latent_dim))# 生成r*c大小的空陣列
    z[0] = np.random.standard_normal((1, latent_dim))# 第一個noise    
    tmp = np.random.standard_normal((1, latent_dim)) - z[0]# 第一個noise與第二個noise差
    for i in range(1, len(z)):
        z[i] = z[i-1] + tmp / (r*c)      
    
    gen_img = decoder.predict(z)
    fig, axs = plt.subplots(r, c, figsize=(10,10))
    cnt = 0
    for j in range(r):
        for i in range(c):                
            axs[j, i].imshow(gen_img[cnt])
            axs[j, i].axis('off')
            cnt = cnt+1
    fig.savefig("images/morphing.png")
    plt.close()

# TODO: 畫出loss圖片
def show(history):
    plt.plot(list(range(len(history))), history)
    plt.ylabel('Loss')
    plt.xlabel('epochs')
    plt.title('Learning curve')
    plt.savefig('images/learning_curve.png')

#%%
# TODO: 設定參數
img_path = './cartoon'
img_rows = 32
img_cols = 32
channels = 3
img_size = (img_rows, img_cols, channels)
#img_data = load_data()          
batch_size = 128
latent_dim = 64
epochs = 10
if not os.path.exists('images/reconst'):
    os.makedirs('images/reconst')
if not os.path.exists('images/generate'):
    os.makedirs('images/generate')

#%%
# TODO: 模型初始化並接起來
img_input = Input(shape=img_size)
encoder = build_encoder()
encoder.summary()
decoder = build_decoder()
decoder.summary()
output = decoder(encoder(img_input)[0])
combined = Model(img_input, output, name='vae')

#%%
# TODO: 計算 reconstruction_loss 與 kl_loss
_, z_mu, z_logvar = encoder.get_output_at(-1)
reconstruction_loss = binary_crossentropy(K.flatten(img_input), K.flatten(output))
reconstruction_loss *= img_size[0] * img_size[1] * img_size[2]

kl_loss = 1 + z_logvar - K.square(z_mu) - K.exp(z_logvar)
kl_loss = K.sum(kl_loss, axis=-1) * -.5
vae_loss = K.mean(reconstruction_loss + 10*kl_loss)
combined.add_loss(vae_loss)
combined.compile(optimizer='adam')

#%%
# TODO: train VAE
History = [] 
for i in range(epochs):
    res = combined.fit(img_data, epochs=10, batch_size=batch_size, verbose=1)
    History += res.history['loss']            
    if i % 50 == 0:
        print("[Epoch %d/%d] [VAE loss: %f]" % (i, epochs, History[-1]))
        sample_image(i)
        generate_image(i)
#%%
show(History)

#%%
r, c = 7, 7
z = np.zeros((r*c, latent_dim))
z[0] = np.random.standard_normal((1, latent_dim))
tmp = np.random.standard_normal((1, latent_dim)) - z[0]
for i in range(1, len(z)):
    z[i] = z[i-1] + tmp / (r*c)      

gen_img = decoder.predict(z)
fig, axs = plt.subplots(r, c, figsize=(10,10))
cnt = 0
for j in range(r):
    for i in range(c):                
        axs[j, i].imshow(gen_img[cnt])
        axs[j, i].axis('off')
        cnt = cnt+1
fig.savefig("images/morphing.png")
plt.close()



