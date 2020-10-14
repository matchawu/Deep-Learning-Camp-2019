# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 22:52:58 2019

@author: Kuo
"""

import gym
import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras import utils as np_utils
from keras.layers import Input, Dense

class utilsToos(object):
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def processReward(r, discount = False, discount_factor = .99):
        # TODO: 計算 Reward
        '''
        對收集到的 reward 進行處理

        Parameters:
            - r (list): 原始收集到的 reward list
            - discount (bool): 是否要對 r 計算 discount
            - discount_factor (float): discount factor
    
        Returns:
            - final_r (np.array): 經處理後的 reward list
       '''
        if discount:
            '''
            discounted reward，計算方式：
                Suppose 
                    1. Discount factor GAMMA = 0.99
                    2. Raw reward R = [r_0, r_1, r_2, r_2]
                    
                Then discounted reward list = [d_0, d_1, d_2, d_3] where
                	d_0 = r_0 + GAMMA*r_1 + GAMMA^2*r_2 + GAMMA^3*r_3
                	d_1 = r_1 + GAMMA*r_2 + GAMMA^2*r_3
                	d_2 = r_2 + GAMMA*r_3
                	d_3 = r_3
            '''
            discounted_r = np.zeros_like(r, dtype=np.float32)
            running_add = 0
            for t in reversed(range(len(r))):
        
                running_add = running_add * discount_factor + r[t]
                discounted_r[t] = running_add
        
            return discounted_r
        else:
            '''
            MC 的 reward 計算方式：
                Suppose
                    1. Raw reward R = [r_0, r_1, r_2, r_2]
                    
                Then the final reward list = [d_0, d_1, d_2, d_3 ] where
                    d_0 = r_0 + r_1 + r_2 + r_3
                	d_1 = r_0 + r_1 + r_2 + r_3
                	d_2 = r_0 + r_1 + r_2 + r_3
                	d_3 = r_0 + r_1 + r_2 + r_3
            '''
            return np.array([np.sum(r)]*len(r))
        
    @staticmethod
    def getRandomReward(env, EPISODE):
        '''
        reward curve 的 baseline，action 是隨機選取
        
        Parameters: 
            - env (gym): 環境
            - EPISODE (int): 回合數
            
        Returns:
            - r (list): 每個回合所得到的總reward
        '''
        random_reward = []
        
        for ep in range(EPISODE):
            _ = env.reset()
            done = False
            reward_counter = 0
            
            while not done:
                action = env.action_space.sample()
                _, reward, done, _ = env.step(action)
                
                reward_counter += reward
                
            random_reward.append(reward_counter)
        
        return random_reward

class Agent():
    '''
    實作 Policy-based : REINFROCE
    '''
    
    def __init__(self, s_dim, a_dim, n, lr, discount, discount_factor):
        '''
        Parameters: 
            - s_dim (int): observation 的維度
            - a_dim (int): action 的維度
            - n (int): 蒐集 n 個 trajectory 才更新 model
            - lr (float): Agent learning rate
            - discount (bool): 是否要計算 discount reward
            - discount_factor (float): discount factor 預設是0.99
        '''
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n = n
        self.lr = lr
        self.discount = discount
        self.discount_factor = discount_factor
        self.__buildModel()
        
        self.s_batch = np.empty(n, dtype=object)
        self.a_batch = np.empty(n, dtype=object)
        self.r_batch = np.empty(n, dtype=object)
        self.batch_counter = 0
        

    def __buildModel(self):
        # TODO: 建立 Agent(只要在input_layer跟output_layer之間加hidden layer就好)
        input_layer = Input(shape=(self.s_dim, ))
        h = Dense(4, activation='selu')(input_layer)
        h = Dense(4, activation='selu')(h)
        output_layer = Dense(self.a_dim, activation='softmax')(h)
        
        self.model = Model(inputs = input_layer, outputs = output_layer)
        self.model.compile(loss = 'categorical_crossentropy', 
                           optimizer = Adam(lr=self.lr))
    
    def sampleAction(self, s):
        # TODO: 讓 Agent sample 一個 action
        act_prob = np.squeeze(self.model.predict(s)) # p只吃一維，但predict出來是(1,2)所以要把它變成一維
        return np.random.choice(np.arange(self.a_dim), p = act_prob)
    
    def storeSample(self, s, a, r):
        # TODO: 儲存一個 episode 中所有的(s,a,r)pair
        a_onehot = np_utils.to_categorical(a, num_classes = self.a_dim)
        r_tmp = utilsToos.processReward(r, self.discount, self.discount_factor)
        
        self.s_batch[self.batch_counter] = np.vstack(s)
        self.a_batch[self.batch_counter] = a_onehot
        self.r_batch[self.batch_counter] = r_tmp
        
        self.batch_counter += 1
    
    def fit(self):
        S = np.concatenate(self.s_batch)
        A = np.concatenate(self.a_batch)
        R = np.concatenate(self.r_batch)
        
        R -= np.mean(R)
        R /= (np.std(R)+K.epsilon())
        R += K.epsilon()
        
#        self.model.fit(S, A*R.reshape(-1, 1), batch_size=64, epochs = 10, verbose = 0)
        self.model.fit(S, A, sample_weight = R, epochs = 1, verbose = 0)
#         
        self.batch_counter = 0
        
# =============================================================================
# Main function
# =============================================================================
if __name__ == '__main__':
    # TODO: 參數設定
    EPISODE = 1000 # 要玩幾個回合
    N = 10 # sample 多少個回合才 update model 一次
    LR = 0.01 # Agent learning rate
    RENDER = False # 是否要顯示遊戲畫面
    DISCOUNT = False # 是否要計算 discount reward
    DISCOUNT_FACTOR = 0.99
    
    '''
    Dear all,
    
        原則上他們在 utilsToos 中的 processReward，只要實作 discount=False 的那段
    就好，所以這個答案有些部分是非必要的，像是上面的超參數：DISCOUNT、DISCOUNT_FACTOR，
    所以我算是把最完整的code給大家了，不一定要求他們寫的跟我的一模一樣。
    '''
    
    env = gym.make('CartPole-v0') # 建立環境
    s_dim = env.observation_space.shape[0] # observation 的維度
    a_dim = env.action_space.n # action 的維度
    
    # 建立 Agent
    agent = Agent(s_dim, a_dim, N, LR, DISCOUNT, DISCOUNT_FACTOR)
    
    # 取得隨機動作的 reward curve
    random_reward = utilsToos.getRandomReward(env, EPISODE)
    
    # 儲存 Agent 的 reward curve
    agent_reward = np.zeros(EPISODE)
    
    # 儲存 trajectory
    s_buffer = np.empty(env.spec.max_episode_steps, dtype=object)
    a_buffer = np.empty(env.spec.max_episode_steps, dtype=object)
    r_buffer = np.empty(env.spec.max_episode_steps, dtype=object)
    
    for ep in range(1, EPISODE+1):
        state = env.reset()
        done = False
        
        buffer_counter = 0
        reward_counter = 0
        
        while not done:
            
            if RENDER: env.render()
            
            action = agent.sampleAction(state[None])
            state2, reward, done, info = env.step(action)
            
            reward_counter += reward
            
            s_buffer[buffer_counter] = state
            a_buffer[buffer_counter] = action
            r_buffer[buffer_counter] = reward
            buffer_counter += 1
            
            state = state2
            
            if done: 
                agent.storeSample(s_buffer[:buffer_counter], 
                                  a_buffer[:buffer_counter], 
                                  r_buffer[:buffer_counter])
            
        agent_reward[ep-1] = reward_counter
        
        if ep % N == 0: 
            agent.fit()
        
        print('[ep. %.4d] rewards = %.4f'%(ep, reward_counter))
        
    # 如果有使用 env.render() 記得要呼叫 env.close() 不然會壞掉
#    env.close() 
    
    # 把 Agent 跟 basleine 的 reward curve 畫出來
    plt.plot(random_reward, label='random choice')
    plt.plot(agent_reward, label='policy gradient')
    plt.title('reward curve')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()
