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
    def processReward(r):
        # TODO: 計算 Reward
        '''
        對收集到的 reward 進行處理

        Parameters:
            - r (list): 原始收集到的 reward list
    
        Returns:
            - final_r (np.array): 經處理後的 reward list
            
        Hint:
            Suppose Raw reward R = [r_0, r_1, r_2, r_2]
            Then the final reward list = [d_0, d_1, d_2, d_3] where
                d_0 = r_0 + r_1 + r_2 + r_3
            	d_1 = r_0 + r_1 + r_2 + r_3
            	d_2 = r_0 + r_1 + r_2 + r_3
            	d_3 = r_0 + r_1 + r_2 + r_3
        '''
        prior = 0
        out = np.zeros_like(r)
        for i in reversed(range(len(r))):
            prior = prior + r[i]
            out[i] = prior
        out = out - np.mean(out)
        out = out / np.std(out)
        
        return out 
#        prior = 0
#        final_r = np.zeros_like(r)
#        for i in range(len(r)):
#            prior = prior + r[i]
#        for i in range(len(r)):    
#            final_r[i] = prior       
#        return final_r
#    
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
                _, reward, done, _ = env.step(env.action_space.sample())
                reward_counter += reward
                
            random_reward.append(reward_counter)
        
        return random_reward
#%%
class Agent():
    '''
    實作 Policy-based Algo.: REINFROCE
    '''
    def __init__(self, s_dim, a_dim, n, lr):
        '''
        Parameters: 
            - s_dim (int): observation 的維度
            - a_dim (int): action 的維度
            - n (int): 蒐集 n 個 trajectory 才更新 model
            - lr (float): Agent learning rate
            
        '''
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n = n
        self.lr = lr
        self.__buildModel()
        
        # 儲存 n 個回合的 sample 用
        self.s_batch = np.empty(n, dtype = object)
        self.a_batch = np.empty(n, dtype = object)
        self.r_batch = np.empty(n, dtype = object)
        
        self.batch_counter = 0
        
    def __buildModel(self):
        # TODO: 建立 Agent(只要在input_layer跟output_layer之間加hidden layer就好)
        input_layer = Input(shape=(self.s_dim, ))
        h = Dense(64, activation='relu')(input_layer)
#        h = Dense(16)(h)
        output_layer = Dense(self.a_dim, activation='softmax')(h)
        self.model = Model(inputs = input_layer, outputs = output_layer)
        self.model.compile(loss = 'categorical_crossentropy', 
                           optimizer = Adam(lr = self.lr))
        self.model.summary()
    
    def sampleAction(self, s):
        action = np.random.choice(a_dim,p=self.model.predict([s])[0])
        return action
        # TODO: 讓 Agent sample 一個 action
        
    
    def storeSample(self, s, a, r):
        a_onehot = np_utils.to_categorical(a,num_classes=self.a_dim)
        r_tmp=utilsToos.processReward(r)
        
        # TODO: 儲存一個 episode 中所有的(s,a,r)pair  
        self.s_batch = np.array([record for record in s])
        # action 獨熱編碼處理，方便求動作概率，即 prob_batch
        self.a_batch = np.array([[1 if record == i else 0 for i in range(2)]
                            for record in a])
        # 假設predict的概率是 [0.3, 0.7]，選擇的動作是 [0, 1]
        # 則動作[0, 1]的概率等於 [0, 0.7] = [0.3, 0.7] * [0, 1]
        self.r_batch = utilsToos.processReward([record for record in r])
#          
    def fit(self):
        # TODO: 用 N 個trajectory 的更新 agent
        # self.model.fit(s, a*r...)
        
        prob_batch = self.model.predict(self.s_batch) * self.a_batch
        self.model.fit(self.s_batch, prob_batch,sample_weight=self.r_batch, verbose=1)
        
if __name__ == '__main__':
    # TODO: 參數設定
    EPISODE = 1500 # 要玩幾個回合
    N = 10 # sample 多少個回合才 update model 一次
    LR = 0.01 # Agent learning rate
    RENDER = 0 # 是否要顯示遊戲畫面
    
    # 建立環境
    env = gym.make('CartPole-v0') 
    s_dim = env.observation_space.shape[0] # observation 的維度
    a_dim = env.action_space.n # action 的維度
    
    # 建立 Agent
    agent = Agent(s_dim, a_dim, N, LR)
    
    # 取得隨機動作的 reward curve
    random_reward = utilsToos.getRandomReward(env, EPISODE)
    
    # 儲存 Agent 的 reward curve
    agent_reward = np.zeros(EPISODE)
    
    # 儲存 trajectory 先開好空的
    s_buffer = np.empty(env.spec.max_episode_steps, dtype = object)
    a_buffer = np.empty(env.spec.max_episode_steps, dtype = object)
    r_buffer = np.empty(env.spec.max_episode_steps, dtype = object)
    
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
            
            if done: #結束了這個t
                agent.storeSample(s_buffer[:buffer_counter], 
                                  a_buffer[:buffer_counter], 
                                  r_buffer[:buffer_counter])
            
        agent_reward[ep-1] = reward_counter
        
        if ep % N == 0: agent.fit()
        print('[ep. %.4d] rewards = %.4f'%(ep, reward_counter))
        
    # 如果有使用 env.render() 記得要呼叫 env.close()
    env.close() 
  #%%  
    # 把 Agent 跟 basleine 的 reward curve 畫出來
    plt.plot(random_reward, label='random choice')
    plt.plot(agent_reward, label='policy gradient')
    plt.title('reward curve')
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.legend()
    plt.show()
