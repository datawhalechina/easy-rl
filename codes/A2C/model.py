#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-05-03 21:38:54
LastEditor: JiangJi
LastEditTime: 2021-05-03 21:40:06
Discription: 
Environment: 
'''
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim):
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