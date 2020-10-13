# -*- coding: utf-8 -*-

#1-1000名pair train
#1001-1500步驟六
#samplelist=np.random.choice(len(index),size=smaple_size)
#index_=index[samplelist]
#samples個3萬筆
#總價元drop掉

index = np.array(list(itertools.combinations(np.arange(1000), 2)))
#combination = (itertools.combinations(np.arange(50), 2))
#index = np.array([[i, j] for i, j in combination])
# Shuffle
np.random.shuffle(index)

#%%
# -*- coding: utf-8 -*-

from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model
import matplotlib.pyplot as plt
from keras import backend
import keras.optimizers
import pandas as pd 
import numpy as np
import itertools
from sklearn import preprocessing
#可能會有warning
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('0711_Landata.csv')
data=data.drop(columns=['Unnamed: 0'])
data = data.sort_values(['總價元'], ascending=[False])
#del data['土地移轉總面積平方公尺']
#%%
data = data[['Rank','房齡','總樓層數','lat','lng','總價元','土地移轉總面積平方公尺','最大移轉樓層','最小移轉樓層']]
data['總價元'] = data['總價元']/10000
#%%
'''01 Data Process'''

# 篩選features 總價除10000

# split train test
train_y = data['總價元'][:1000].reset_index(drop=True)
test_y = data['總價元'][1000:1500].reset_index(drop=True)


train_x = data[:1000].drop(columns=['Rank', '總價元']).reset_index(drop=True)
test_x = data[1000:1500].drop(columns=['Rank', '總價元']).reset_index(drop=True)


train_x = np.array(train_x)
train_y = np.array(train_y) 
test_x = np.array(test_x)
test_y = np.array(test_y)
#%%
'''02 Model Build: RankNet'''
from keras.models import Sequential, Model
from keras.layers import Dense, Subtract, Activation, Input

feature_num = 7
marker = Sequential()
marker.add(Dense(64, activation='selu', input_shape=(feature_num,)))
marker.add(Dense(32, activation='selu'))
marker.add(Dense(16, activation='selu'))
marker.add(Dense(4, activation='selu'))
marker.add(Dense(1))

i_input = Input(shape=(feature_num,))
#j_input = Input(shape=(feature_num,))
i_score = marker(i_input)
#j_score = marker(j_input)

##diff = Subtract()([i_score, j_score])
#output = Activation('sigmoid')(i_score)

ranknet = Model([i_input], i_score)
#optimizer = keras.optimizers.Adam(lr=1e-3, decay=0.0) #lr設learning rate decay
ranknet.summary()
ranknet.compile(optimizer = 'adam', loss = "mse",metrics=["accuracy"])

#%%
'''03 PairWise: Making pairs'''
# Pairs
#index_train = np.array(list(itertools.permutations(np.arange(1000), 2)))
#index_test = np.array(list(itertools.permutations(np.arange(500), 2)))
#combination = (itertools.combinations(np.arange(50), 2))
#index = np.array([[i, j] for i, j in combination])

# Shuffle
#np.random.shuffle(index_train)
#np.random.shuffle(index_test)

#samplesize = 10000
#samplelist = np.randim.choice(len(index),size=samplesize)
#index_ = index[samplelist]

#import random
#
#for i in range(len(index_train)):
#    is_shuffle=random.randint(0,1)
#    if is_shuffle :
#        tems=index_train[i][0]
#        index_train[i][0]=index_train[i][1]
#        index_train[i][1]=tems
#    


'''04 Making target'''
#def target(I, J):
#    target = np.subtract(I, J) 
#    target = np.clip(np.sign(target), 0, 1)
#    return target
#
#target_train = target(train_y[index_train[:,0]], train_y[index_train[:,1]])
#target_test = target(test_y[index_test[:,0]], test_y[index_test[:,1]])

#%%
'''05 Training'''
# Trianing 
EPOCHS = 300
BATCH_SIZE = 200

res = ranknet.fit(train_x, train_y, \
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, \
                         validation_data=(test_x, test_y))
#aa=index_train[:,0]
'''Plot Loss'''
plt.plot(res.history['loss'], label='train')
plt.plot(res.history['val_loss'], label='test')
plt.title('Loss')
plt.legend()
plt.show()
plt.close()

'''Plot Accuracy'''
plt.plot(res.history['acc'], label='train')
plt.plot(res.history['val_acc'], label='test')
plt.title('Accuracy')
plt.legend()
plt.show()
plt.close()
#%%
'''06 Predict''' #1001-1010
ans = test_y
score = np.array(marker.predict(test_x)).reshape(-1,)
test_xp= pd.Series(test_y)
temp = test_xp.reset_index(drop=True)
Top_1001 = temp[np.argsort(score, axis=0)[::-1]][:100]
#Top_10 = temp[np.argsort(score, axis=0)[::-1]][:10]

Rec1 = len(set(Top_1001) & set(ans[:100]))
#Rec10 = len(set(Top_10) & set(ans[:10]))
print('Rec1：%d' % (Rec1))
#%%
marker.predict
#%%
'''大絕招'''
model.load_weights('0710LTR_osu!_5')