#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 21:14:12
LastEditor: John
LastEditTime: 2022-08-29 14:24:44
Discription: 
Environment: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的特征数即环境的状态维度
            output_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ActorSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorSoftmax, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self,state):
        dist = F.relu(self.fc1(state))
        dist = F.softmax(self.fc2(dist),dim=1)
        return dist
class Critic(nn.Module):
    def __init__(self,input_dim,output_dim,hidden_dim=256):
        super(Critic,self).__init__()
        assert output_dim == 1 # critic must output a single value
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self,state):
        value = F.relu(self.fc1(state))
        value = self.fc2(value)
        return value

class ActorCriticSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, actor_hidden_dim=256,critic_hidden_dim=256):
        super(ActorCriticSoftmax, self).__init__()

        self.critic_fc1 = nn.Linear(input_dim, critic_hidden_dim)
        self.critic_fc2 = nn.Linear(critic_hidden_dim, 1)

        self.actor_fc1 = nn.Linear(input_dim, actor_hidden_dim)
        self.actor_fc2 = nn.Linear(actor_hidden_dim, output_dim)
    
    def forward(self, state):
        # state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        value = F.relu(self.critic_fc1(state))
        value = self.critic_fc2(value)
        
        policy_dist = F.relu(self.actor_fc1(state))
        policy_dist = F.softmax(self.actor_fc2(policy_dist), dim=1)

        return value, policy_dist

class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value