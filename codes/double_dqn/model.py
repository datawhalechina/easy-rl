#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:47:02
@LastEditor: John
LastEditTime: 2020-08-19 16:55:54
@Discription: 
@Environment: python 3.7.7
'''
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self, n_states=4, n_actions=18):
        """ 初始化q网络，为全连接网络
            n_states: 输入的feature即环境的state数目
            n_actions: 输出的action总个数
        """
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(n_states, 128) # 输入层
        self.fc2 = nn.Linear(128, 128) # 隐藏层
        self.fc3 = nn.Linear(128, n_actions) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)