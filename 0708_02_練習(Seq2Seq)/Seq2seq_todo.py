# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 21:53:02 2019

@author: MUII SEE
"""

#TODO: IMPORT你會用到的套件
import pandas as pd
import numpy as np
from keras.layers import LSTM,Dense,Input
from keras.models import Model 

#%%
#TODO: 挑選最合適的參數
batch_size = 64           # Batch size for training.
epochs =   50            # Number of epochs to train for.
latent_dim =  256           # Latent dimensionality of the encoding space.
num_samples =  10000           # Number of samples to train on.


#%%
#TODO:讀取資料
#Hint: encoding = 'utf8'
data = pd.read_csv('cmn.txt',sep="	", header=None, encoding = 'utf8')[:num_samples]
data.columns = ["E", "C"]

#%%
#TODO:這裡將raw data分別切割成中文句子、英文句子、中文字（含空白格、符號，取不重覆）、英文字母（含空白格、符號，取不重覆）
#Hint:利用'\t' split出英文句子， 利用'\t'和'\n' split出中文句子(當然你也可以用自己的方法進行處理)
# Vectorize the data.
input_texts =  data['E']           #英文句子
target_texts =  data['C']          #中文句子
input_characters =  []      #取英文字母，含空白格、符號，不重覆
target_characters =  []     #取中文字，含空白格、符號，不重覆

for i in range(num_samples):
    temp = input_texts[i]
    for j in temp:
        input_characters.append(j)
        
        
for p in range(num_samples):
    temp = target_texts[p]
    for q in temp:
        target_characters.append(q)
#Hint:這裡需要寫for迴圈取出中文句子、英文句子、中文字（含空白格、符號，取不重覆 !!注意!!: target texts也會出現英文字母也要算進去）、英文字母（含空白格、符號，取不重覆）
#當然你也可以用自己的方法來取～


se = set(input_characters)
dictionary_e = {e:i for i,e in enumerate(se)}
dictionary_e = sorted(list(dictionary_e))

sc = set(target_characters)
dictionary_c = {e:i for i,e in enumerate(sc)}
dictionary_c = sorted(list(dictionary_c))
# We use "tab" as the "start sequence" character
# for the targets, and "\n" as "end sequence" character.


#%%
#TODO:處理相關變數
input_characters =  sorted(list(input_characters))                      #對取英文字母進行排序
target_characters =  sorted(list(target_characters))                            #對取中文字進行排序
num_encoder_tokens = len(dictionary_e)                     #計算英文字母的數量
num_decoder_tokens =  len(dictionary_c)                    #計算中文字的數量
max_encoder_seq_length =  max([len(t) for t in input_texts])            #取在英文句子中字數最多的句子
max_decoder_seq_length =  max([len(t) for t in target_texts])               #取出在中文句子中字數最多的句子

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


#%%
#TODO: 對英文字母（input_characters）以及 中文字（target_characters）建立字典
#Example: input_token_index = { z : 71 }...
#Example 2:  target_token_index = {人 : 99}...
#I'm biiiiiiig hint: 可以拔上週RNN練習建字典的方法過來～
input_token_index =  dict([(char,i)for i, char in enumerate(dictionary_e)])         #英文字典
target_token_index =  dict([(char,i)for i, char in enumerate(dictionary_c)])         #中文字典

#%%
# 設定編碼器、解碼器input起始值(均為0矩陣)
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

# 設定 encoder_input、decoder_input對應的順序 
for i, (input_text, target_text) in enumerate(zip(input_texts[:10000], target_texts[:10000])):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


#%%
###開始建利用LSTM來建我們的Encoder
            
# 建立 encoder LSTM 隱藏層
# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens)) #input都必須是英文句子相關的data，這裡宣告了input shape的大小
encoder = LSTM(latent_dim, return_state=True)            #定義encoder layers
encoder_outputs, state_h, state_c = encoder(encoder_inputs) 

# 捨棄 output，只保留記憶狀態 h 及 c
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

#%%
###開始建利用LSTM來建我們的Decoder
# 建立 decoder LSTM 隱藏層
#TODO: 定義decoder的Input
#Hint: 參照encoder_inputs的寫法
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder = LSTM(latent_dim, return_sequences = True,return_state=True)
decoder_outputs, _, _ = decoder(decoder_inputs, initial_state = encoder_states) 


# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
# decoder 記憶狀態不會在訓練過程使用，只會在推論(Inference)使用

#TODO：使用LSTM來搭建Decoder層， 注意這裡記得讓它開啟return_sequences
#Hint: 參照encoder的做法


#TODO: 使用剛剛定義好的decoder_lstm
#Hint: 可以參照上面encoder_outputs, state_h, state_c 的寫法，同時使用 `encoder_states` 作為 initial state。


#TODO: 定義decoder dense層 node的數量，以及使用到的Activation function
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)


#%%
# TODO: 定義模型，由 encoder_input_data 及 decoder_input_data 轉換為 decoder_target_data 
# Define the model that will turn `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
# I'm Biiiiiiiiiig Hint: --- [Example] model = Model([i1, i2], o1)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)



#%%

#TODO: 選擇你的optimizer、loss, 然後開始training
#Hint: model.fit 會有encoder和decoder兩個input，以及decoder一個output
#COMPILE
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#fit
hist = model.fit([encoder_input_data,decoder_input_data],decoder_target_data,
          batch_size,
          epochs,
          validation_split=0.2,
          verbose=1)



#%%
# 推論(Inference)
# 過程如下:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states
# Define sampling models

# 定義編碼器取樣模型
# TODO: 定義編碼器取樣模型，由 encoder_inputs 及 encoder_states 
# Hint: 參照上面model的寫法，注意：這裡用到的只有input以及states
encoder_model = Model(encoder_inputs,encoder_states)

#%%
# 定義解碼器的input
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))

#TODO: decoder_states_inputs由input h以及input c所組成
#Hint: 可以參照encoder_states的寫法，但這裡使用到decoder相關的h以及c
decoder_states_inputs = [decoder_state_input_h,decoder_state_input_c]

# 定義解碼器 LSTM 模型
decoder_outputs, state_h, state_c = decoder(
    decoder_inputs, initial_state=decoder_states_inputs)

#TODO:以編碼器的記憶狀態 h 及 c 為解碼器的記憶狀態 
#Hint: 可以參照encoder_states的寫法
decoder_states = [state_h,state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

#%%
# Reverse-lookup token index to decode sequences back to something readable.
#TODO: 建立反向的 dict，才能透過查詢將數值轉回文字

reverse_input_char_index = {v: k for k, v in input_token_index.items()}
reverse_target_char_index = {v: k for k, v in target_token_index.items()}


#%%
# 模型預測，並取得翻譯結果(中文)    
def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index[' ']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence

# 測試100次
for seq_index in range(100):
    # Take one sequence (part of the training test)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('*')
    print('Input sentence:', input_texts[seq_index])
    try:
        print('Decoded sentence:', decoded_sentence)
    except:
        print('Decoded sentence:', decoded_sentence.encode('ascii', 'replace'))
        #print("error:", sys.exc_info()[0])