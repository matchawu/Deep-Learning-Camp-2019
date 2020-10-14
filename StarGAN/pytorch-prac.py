# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:40:30 2019

@author: wwj
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from keras.datasets import mnist

DEVICE = torch.device('cuda:0') #using GPU
#%%

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(28*28,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,16)
        self.output = nn.Linear(16,10)
        
    def forward(self,x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.output(h)
    
#%%
#compile  
#classifier = LeNet()
classifier = LeNet().to(DEVICE)
#loss
ce_criterion = nn.CrossEntropyLoss()
#optimizer
optimizer = torch.optim.Adam(classifier.parameters(),lr=.001)

(train_x,train_y),(test_x,test_y) = mnist.load_data()
train_x = train_x.reshape((-1,28*28))/255
test_x = test_x.reshape((-1,28*28))/255


#train
BATCH_SIZE = 256
test_x = torch.from_numpy(test_x).type('torch.FloatTensor').to(DEVICE)
test_y = torch.from_numpy(test_y).type('torch.LongTensor').to(DEVICE)
for e in range(20):
    for b in range(0, len(train_x), BATCH_SIZE):
        #使得可以調參數
        classifier.train()
        
        x,y = train_x[b:b+BATCH_SIZE], train_y[b:b+BATCH_SIZE]
        x = torch.from_numpy(x).type('torch.FloatTensor').to(DEVICE)
        y = torch.from_numpy(y).type('torch.LongTensor').to(DEVICE)
        
        optimizer.zero_grad() #歸零 否則微分量會累加
        pred_y = classifier(x)
        loss = ce_criterion(pred_y,y) #可以任意加東西 自己定義loss acc
        acc = torch.sum(torch.argmax(pred_y, dim=1)==y).item() / len(y)
        #backpropagation
        loss.backward()
        #optimizer做更新
        optimizer.step()
        
        classifier.eval() #參數不會變
        with torch.no_grad():
            eval_y = classifier(test_x)
            test_loss = ce_criterion(eval_y, test_y)
            test_acc = torch.sum(torch.argmax(eval_y, dim=1)==test_y).item() /len(y)
    print('[train_loss: %.6f, test_loss: %.6f, train_acc: %.6f, test_acc: %.6f]'% (loss.item(), test_loss.item(), acc, test_acc))
            
        
        
        