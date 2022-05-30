#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:27:44
LastEditor: John
LastEditTime: 2022-02-10 01:25:27
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
import numpy as np

class MLP(nn.Module):
    
    ''' 多层感知机
        输入：state维度
        输出：概率
    '''
    def __init__(self,input_dim,hidden_dim = 36):
        super(MLP, self).__init__()
        # 24和36为hidden layer的层数，可根据input_dim, n_actions的情况来改变
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x
        
class PolicyGradient:
    
    def __init__(self, n_states,cfg):
        self.gamma = cfg.gamma
        self.policy_net = MLP(n_states,hidden_dim=cfg.hidden_dim)
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=cfg.lr)
        self.batch_size = cfg.batch_size

    def choose_action(self,state):
        
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net(state)
        m = Bernoulli(probs) # 伯努利分布
        action = m.sample()
        action = action.data.numpy().astype(int)[0] # 转为标量
        return action
        
    def update(self,reward_pool,state_pool,action_pool):
        # Discount reward
        running_add = 0
        for i in reversed(range(len(reward_pool))):
            if reward_pool[i] == 0:
                running_add = 0
            else:
                running_add = running_add * self.gamma + reward_pool[i]
                reward_pool[i] = running_add

        # Normalize reward
        reward_mean = np.mean(reward_pool)
        reward_std = np.std(reward_pool)
        for i in range(len(reward_pool)):
            reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

        # Gradient Desent
        self.optimizer.zero_grad()

        for i in range(len(reward_pool)):
            state = state_pool[i]
            action = Variable(torch.FloatTensor([action_pool[i]]))
            reward = reward_pool[i]
            state = Variable(torch.from_numpy(state).float())
            probs = self.policy_net(state)
            m = Bernoulli(probs)
            loss = -m.log_prob(action) * reward  # Negtive score function x reward
            # print(loss)
            loss.backward()
        self.optimizer.step()
    def save(self,path):
        torch.save(self.policy_net.state_dict(), path+'pg_checkpoint.pt')
    def load(self,path):
        self.policy_net.load_state_dict(torch.load(path+'pg_checkpoint.pt')) 