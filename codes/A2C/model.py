#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-03 20:45:25
LastEditor: John
LastEditTime: 2021-03-20 17:41:33
Discription: 
Environment: 
'''
import torch.nn as nn
from torch.distributions import Categorical

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
        print(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value