# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 16:23:18 2019

@author: Susan
"""
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
#del data[0]


#rank = data['Rank']
unnamed = data['Unnamed: 0']
del data['Unnamed: 0']

# save player name
player = data['player_name']

# get dummies (device)
#data= pd.concat([data, pd.get_dummies(data['device']).astype(int)], axis=1).drop(columns=['device', 'player_name'])
#%%
'''01 Data Process'''
'''Columns
# 'rank', 'country_rank', 'player_name', 'country', 'accuracy','play_count', 'level',\
  'hours', 'performance_points', 'ranked_score', 'ss', 's', 'a', 'watched_by', 'total_hits', 'device'
'''

# split train test
train_y = data[1::2]['performance_points'].reset_index(drop=True)
train_x = data[1::2].drop(columns=['rank', 'performance_points']).reset_index(drop=True)

test_y = data[0::2]['performance_points'].reset_index(drop=True)
test_x = data[0::2].drop(columns=['rank', 'performance_points']).reset_index(drop=True)

# data:-100 / 5
train_x['level'] = (train_x['level'] - 100) / max(train_x['level'])
test_x['level'] = (test_x['level'] - 100) / max(train_x['level'])

# accuracy: / 100
train_x['accuracy'] = train_x['accuracy'] / 100
test_x['accuracy'] = test_x['accuracy'] / 100

# country_rank / max
train_x['country_rank'] = train_x['country_rank'] / max(train_x['country_rank'])
test_x['country_rank'] = test_x['country_rank'] / max(train_x['country_rank'])

# tokenize country
#  Making dictionary from training data
word_list = set(train_x['country'])
dictionary = {e:i+1 for i, e in enumerate(word_list)}
dictionary['others'] = 0
for i in range(len(train_x)):
    key = train_x['country'].iloc[i]
    if key in dictionary:
        train_x['country'].iloc[i] = dictionary[key] / len(dictionary)
    else:
        train_x['country'].iloc[i] = 0
for i in range(len(test_x)):
    key = test_x['country'].iloc[i]
    if key in dictionary:
        test_x['country'].iloc[i] = dictionary[key] / len(dictionary)
    else:
        test_x['country'].iloc[i] = 0


# min max 
b = ['hours', 'play_count', 'ranked_score']
temp = train_x[b].values 
scaler = preprocessing.MinMaxScaler().fit(temp)
train_x[b] = scaler.transform(train_x[b].values)
test_x[b] = scaler.transform(test_x[b].values) # transform to test dataset

# z-score
b = ['ss', 's', 'a', 'watched_by', 'total_hits']
temp = train_x[b].values 
scaler = preprocessing.StandardScaler().fit(temp)
train_x[b] = scaler.transform(train_x[b].values)
test_x[b] = scaler.transform(test_x[b].values) # transform to test dataset

train_x['country'], test_x['country'] = train_x['country'].astype(float), test_x['country'].astype(float)

train_x = np.array(train_x)
train_y = np.array(train_y) 
test_x = np.array(test_x)
test_y = np.array(test_y)
#%%
'''02 Model Build: RankNet'''
INPUT_DIM = train_x.shape[1]
# Model.
h_1 = Dense(8, activation = "selu", name = 'h1')
h_2 = Dense(4, activation = "selu", name = 'h2')
h_3 = Dense(2, activation = "selu", name = 'h3')
s = Dense(1) #linear

# Player i
PI = Input(shape = (INPUT_DIM, ), dtype = "float32")
I = h_1(PI)
I = h_2(I)
I = h_3(I)
I_score = s(I)

# player j
PJ = Input(shape = (INPUT_DIM, ), dtype = "float32")
J = h_1(PJ)
J = h_2(J)
J = h_3(J)
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
index = np.array(list(itertools.permutations(np.arange(1000), 2)))
#permutation = itertools.permutations(np.arange(1000), 2)
#index = np.array([[i, j] for i, j in permutation])
#combination = itertools.combinations(np.arange(50), 2)
#index = np.array([[i, j] for i, j in combination])

# Shuffle
np.random.shuffle(index)

samplesize = 10000
samplelist = np.randim.choice(len(index),size=samplesize)
index_ = index[samplelist]

'''04 Making target'''
def target(I, J):
    target = np.subtract(I, J) 
    target = np.clip(np.sign(target), 0, 1) #<0:-1 =0:0 >0:1 if use rank: -np.sign
    return target

target_train = target(train_y[index[:,0]], train_y[index[:,1]]) #以train_y取index
target_test = target(test_y[index[:,0]], test_y[index[:,1]])

#%%
'''05 Training'''
# Trianing 
EPOCHS = 10
BATCH_SIZE = 8

res = model.fit([train_x[index[:,0]], train_x[index[:,1]]], target_train, \
                         epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, \
                         validation_data=([test_x[index[:,0]], test_x[index[:,1]]], target_test))

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
ans = player[0::2][:10]
score = np.array(get_score([test_x])).reshape(-1,)
temp = player[0::2].reset_index(drop=True)
Top_10 = temp[np.argsort(score, axis=0)[::-1]][:10] #argsort:從小到大，::-1整個reverse

Rec = len(set(Top_10) & set(ans[:10]))
print('Rec：%d' % (Rec))
#%%
'''大絕招'''
model.load_weights('0710LTR_osu!_5')