
# coding: utf-8

# In[1]:


import tensorflow as tf
import os 

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
config = tf.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)


# In[2]:


import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os

from keras.losses import binary_crossentropy
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, Dense, Input, BatchNormalization, Flatten, Lambda, Reshape, LeakyReLU, ReLU


# In[3]:


class VAE:
    def __init__(latent_dim, img_path='./cartoon'):
        img_rows = 64
        img_cols = 64
        channels = 3
        img_size = (img_rows, img_cols, channels)
        latent_dim = 100
        img_path_list = glob(img_path + '/*/*.png')
        img_data = load_data(img_size, img_path_list)
    
    def load_data( img_res, path_list):
        def imread(path, img_res):           
            return np.array(Image.open(path).convert('RGB').resize(img_res, Image.ANTIALIAS)) / 255
        
        img_set = np.zeros((len(path_list), ) + img_res)
        for idx, rand_img_path in enumerate(path_list):
            img_set[idx] = imread(rand_img_path, img_res[:2])[:,:,:3]
        return img_set
    
    def build_encoder():
        def sampling(args):
            z_mu, z_logvar = args
            batch = K.shape(z_mu)[0]
            dim = K.int_shape(z_mu)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mu + K.exp(0.5 * z_logvar) * epsilon
        
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

    def build_decoder(self):
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
        
        outputs = Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='sigmoid', name='convT5')(h)

        return Model(latent_inputs, outputs, name='decoder')
    
    def connect():
        img_input = Input(shape=img_size)
        encoder = build_encoder()
        #self.encoder.summary()
        decoder = build_decoder()
        #self.decoder.summary()
        output = decoder(encoder(img_input)[0])
        
        combined = Model(img_input, output, name='vae')
        
        _, z_mu, z_logvar = encoder.get_output_at(-1)
        reconstruction_loss = binary_crossentropy(K.flatten(img_input), K.flatten(output))
        reconstruction_loss *= img_size[0] * img_size[1] * img_size[2]
        
        kl_loss = 1 + z_logvar - K.square(z_mu) - K.exp(z_logvar)
        kl_loss = K.sum(kl_loss, axis=-1) * -.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        combined.add_loss(vae_loss)
        combined.compile(optimizer='adam')
        
    def sample_image( iteration):
        r, c = 4, 3
        idx = np.random.choice(len(img_data), r*c, replace=False)
        y = combined.predict(img_data[idx])

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
    
    def generate_image( iteration):
        r, c = 4, 4
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
        
    def generate_morphing_image():
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
    
    def show():
        plt.plot(list(range(len(history))), history)
        plt.ylabel('Loss')
        plt.xlabel('epochs')
        plt.title('Learning curve')
        plt.savefig('images/learning_curve.png')

    def train( epochs, batch_size):
        history = [] 
        for i in range(epochs):
            history = combined.fit(img_data, epochs=1, batch_size=batch_size, verbose=0)
            history += history.history['loss']            
            if i % 50 == 0:
                print("[Epoch %d/%d] [VAE loss: %f]" % (i, epochs, history[-1]))
                sample_image(i)
                generate_image(i)


# In[4]:


batch_size = 100
latent_dim = 50
epochs = 10
if not os.path.exists('images/reconst'):
    os.makedirs('images/reconst')
if not os.path.exists('images/generate'):
    os.makedirs('images/generate')


# In[5]:


vae = VAE()
vae.connect()
vae.train(epochs, batch_size)
vae.show()


# In[6]:


vae.generate_morphing_image()

