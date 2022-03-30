#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-23 16:35:58
LastEditor: John
LastEditTime: 2021-12-21 23:21:26
Discription: 
Environment: 
'''
import torch.nn as nn
import torch.nn.functional as F
class MLP(nn.Module):
    
    ''' 多层感知机
        输入：state维度
        输出：概率
    '''
    def __init__(self,input_dim,hidden_dim = 36):
        super(MLP, self).__init__()
        # 24和36为hidden layer的层数，可根据input_dim, action_dim的情况来改变
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim,hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc3(x))
        return x