#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-23 15:29:24
LastEditor: John
LastEditTime: 2021-04-08 22:36:43
Discription: 
Environment: 
'''
import torch.nn as nn
from torch.distributions.categorical import Categorical
class Actor(nn.Module):
    def __init__(self,n_states, n_actions,
            hidden_dim):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
                nn.Linear(n_states, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, n_actions),
                nn.Softmax(dim=-1)
        )
    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

class Critic(nn.Module):
    def __init__(self, n_states,hidden_dim):
        super(Critic, self).__init__()
        self.critic = nn.Sequential(
                nn.Linear(n_states, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
        )
    def forward(self, state):
        value = self.critic(state)
        return value