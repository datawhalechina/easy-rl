#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-24 22:14:12
LastEditor: John
LastEditTime: 2021-03-24 22:17:09
Discription: 
Environment: 
'''
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    def __init__(self, state_dim,action_dim,hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim) 
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)