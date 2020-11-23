#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-11-22 23:18:46
LastEditor: John
LastEditTime: 2020-11-23 01:58:22
Discription: 
Environment: 
'''
import torch.nn as nn
import torch.nn.functional as F
class FCN(nn.Module):
    ''' 全连接网络'''
    def __init__(self,n_states):
        super(FCN, self).__init__()
        # 24和36为hidden layer的层数，可根据n_states, n_actions的情况来改变
        self.fc1 = nn.Linear(n_states, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x