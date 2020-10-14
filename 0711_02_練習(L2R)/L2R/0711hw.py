# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 19:04:17 2019

@author: wwj
"""
#%%
#取前10000筆做pairwise ranking
#10000~20000筆做validation 做predict
from keras.layers import Activation, Dense, Input, Subtract
from keras.models import Model
import matplotlib.pyplot as plt
from keras import backend
import keras.optimizers
import pandas as pd 
import numpy as np
import itertools
#from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

#data = pd.read_csv('ranking.csv')
data = pd.read_csv('0711_Landata.csv')
unnamed = data['Unnamed: 0']
#unnamed = unnamed[1000:1500].reset_index(drop=True)
#unnamed = unnamed - 1000
del data['Unnamed: 0']
data = data.sort_values(['Rank'], ascending=[True])
#del data[0]
#rank 1~1000 做pairwise
#1001~1500 做predict

#%%
'''01 Data Process'''
rank = data['Rank']

#del data['Rank']
price =  data['總價元']
price = price[1000:1500]

# split train test
train_y = data[:1000]['總價元'].reset_index(drop=True)
train_x = data[:1000].drop(columns=['Rank', '總價元']).reset_index(drop=True)

test_y = data[1000:1500]['總價元'].reset_index(drop=True)
test_x = data[1000:1500].drop(columns=['Rank', '總價元']).reset_index(drop=True)

train_x = np.array(train_x)
train_y = np.array(train_y) 
test_x = np.array(test_x)
test_y = np.array(test_y)
#%%
'''02 Model Build: RankNet'''
INPUT_DIM = train_x.shape[1]
# Model.
h_1 = Dense(512, activation = "selu", name = 'h1')
h_2 = Dense(128, activation = "selu", name = 'h2')
h_3 = Dense(64, activation = "selu", name = 'h3')
h_4 = Dense(64, activation = "selu", name = 'h4')
h_5 = Dense(64, activation = "selu", name = 'h5')
h_6 = Dense(64, activation = "selu", name = 'h6')
s = Dense(1) #linear

# Player i
PI = Input(shape = (INPUT_DIM, ), dtype = "float32")
I = h_1(PI)
I = h_2(I)
I = h_3(I)
I = h_4(I)
I = h_5(I)
I = h_6(I)
I_score = s(I)

# player j
PJ = Input(shape = (INPUT_DIM, ), dtype = "float32")
J = h_1(PJ)
J = h_2(J)
J = h_3(J)
J = h_4(J)
J = h_5(J)
J = h_6(J)
J_score = s(J)

diff = Subtract()([I_score, J_score])
prob = Activation("sigmoid")(diff)
model = Model(inputs = [PI, PJ], outputs = prob)
get_score = backend.function([PI], [I_score])
optimizer = keras.optimizers.Adam(lr=1e-3, decay=0.0) #lr:learning rate
model.summary()
model.compile(optimizer = optimizer, loss = "binary_crossentropy",metrics=["accuracy"])

#%%
'''03 PairWise: Making pairs'''
# Pairs
index_train = np.array(list(itertools.combinations(np.arange(1000), 2)))
index_test = np.array(list(itertools.combinations(np.arange(500), 2)))
#permutation = itertools.permutations(np.arange(1000), 2)
#index = np.array([[i, j] for i, j in permutation])
#combination = itertools.combinations(np.arange(50), 2)
#index = np.array([[i, j] for i, j in combination])

# Shuffle
np.random.shuffle(index_train)
np.random.shuffle(index_test)

#samplesize = 10000
#samplelist = np.random.choice(len(index_train),size=samplesize)
#index_ = index_train[samplelist]

#%%
'''04 Making target'''
def target(I, J):
    target = np.subtract(I, J) 
    target = np.clip(np.sign(target), 0, 1) #<0:-1 =0:0 >0:1 if use rank: -np.sign
    return target

target_train = target(train_y[index_train[:,0]], train_y[index_train[:,1]]) #以train_y取index
target_test = target(test_y[index_test[:,0]], test_y[index_test[:,1]])

#%%
'''05 Training'''
# Trianing 
EPOCHS = 10
BATCH_SIZE = 8

res = model.fit([train_x[index_train[:,0]], train_x[index_train[:,1]]], target_train, \
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, \
                         validation_data=([test_x[index_test[:,0]], test_x[index_test[:,1]]], target_test))

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
'''06 Predict'''
#1001~1500 predict
ans = unnamed[:][:10]
score = np.array(get_score([test_x])).reshape(-1,)
#test_x = pd.Series(test_y)
temp =  unnamed[:].reset_index(drop=True)
Top_10 = temp[np.argsort(score, axis=0)[::-1]][:10] #argsort:從小到大，::-1整個reverse

#test_xp= pd.Series(test_y)
#temp = test_xp.reset_index(drop=True)
#Top_10 = temp[np.argsort(score, axis=0)[::-1]][:10]
Rec = len(set(Top_10) & set(ans[:10]))
print('Rec：%d' % (Rec))






